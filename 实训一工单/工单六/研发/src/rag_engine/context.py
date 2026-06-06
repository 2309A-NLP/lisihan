# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.config import Config
from src.document import Document

class ContextMixin:
    def _trim_context_part(self, content: str) -> str:
        max_chars = max(300, int(getattr(Config, "LLM_CONTEXT_MAX_CHARS", 1800) / max(Config.TOP_K_RETRIEVAL, 1)))
        text = re.sub(r"\s+", " ", content or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    def _retrieval_debug_preview(self, source_chunks: List[Dict], limit: int = 5) -> List[Dict]:
        previews = []
        for idx, chunk in enumerate(source_chunks[:limit], start=1):
            metadata = chunk.get("metadata", {}) or {}
            content = re.sub(r"\s+", " ", chunk.get("content", "") or "").strip()
            previews.append(
                {
                    "rank": idx,
                    "source_file": metadata.get("source_file", ""),
                    "page": metadata.get("page", 0),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "score": chunk.get("relevance_score", 0.0),
                    "preview": content[:500],
                }
            )
        return previews

    def _direct_index_fallback(self, question: str, source_file: str | None, top_k: int):
        q = question or ""
        fallback_specs = []
        if "军用领域" in q and "收入" in q and not any(term in q for term in ["比重", "占比", "比例", "百分比"]):
            fallback_specs.append(("来自军用领域的收入分别为", "主营业务收入比重分别为"))
        if not fallback_specs:
            return []

        docs = []
        for item in getattr(self.vector_store, "documents", []) or []:
            doc = getattr(item, "doc", None)
            if doc is None:
                continue
            metadata = doc.metadata or {}
            if source_file and metadata.get("source_file") != source_file:
                continue
            content = doc.page_content or ""
            for required, optional in fallback_specs:
                if required in content and (not optional or optional in content):
                    docs.append((doc, 1.0))
                    break
        return docs[:top_k]

    def _build_context(
        self,
        question: str,
        top_k: int = 4,
        question_id: int = None,
        retrieval_mode: str = "hybrid",
        retrieval_config: Dict | None = None,
    ):
        analysis = self.llm_engine.query_intent(question, question_id=question_id)
        search_query = self.rewrite_query(question)
        source_file = self.select_pdf(question)
        if search_query != question:
            analysis["rewritten_query"] = search_query
        analysis["source_file"] = source_file
        retrieved_results = self.vector_store.search(
            search_query,
            top_k,
            mode=retrieval_mode,
            source_file=source_file,
            retrieval_config=retrieval_config,
        )
        retrieval_query_used = search_query
        if not retrieved_results and search_query != question:
            fallback_results = self.vector_store.search(
                question,
                top_k,
                mode=retrieval_mode,
                source_file=source_file,
                retrieval_config=retrieval_config,
            )
            if fallback_results:
                retrieved_results = fallback_results
                retrieval_query_used = question
                analysis["rewrite_fallback_used"] = True
                analysis["rewrite_fallback_query"] = question
        if not retrieved_results:
            direct_results = self._direct_index_fallback(question, source_file, top_k)
            if direct_results:
                retrieved_results = direct_results
                retrieval_query_used = question
                analysis["direct_index_fallback_used"] = True
        if hasattr(self.vector_store, "last_weights"):
            analysis["bm25_weight"] = self.vector_store.last_weights.get("bm25", 0.5)
            analysis["vector_weight"] = self.vector_store.last_weights.get("vector", 0.5)

        context_parts = []
        scores = []
        source_chunks = []
        for doc, score in retrieved_results:
            metadata = doc.metadata or {}
            context_parts.append(self._trim_context_part(doc.page_content))
            scores.append(score)
            source_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": metadata,
                    "relevance_score": score,
                    "auto_fallback": bool(metadata.get("auto_fallback")),
                    "fallback_from": metadata.get("fallback_from", ""),
                }
            )

        debug_preview = self._retrieval_debug_preview(source_chunks)
        auto_fallback_used = any(chunk.get("auto_fallback") for chunk in source_chunks)
        analysis["retrieved_count"] = len(source_chunks)
        analysis["retrieved_debug_preview"] = debug_preview
        analysis["auto_fallback"] = auto_fallback_used
        if auto_fallback_used:
            analysis["fallback_notice"] = "精确匹配无结果，已自动使用模糊匹配"
        for item in debug_preview:
            self.logger.info(
                "retrieved raw chunk | query=%s | source=%s | page=%s | chunk=%s | score=%.4f | preview=%s",
                retrieval_query_used,
                item["source_file"],
                item["page"],
                item["chunk_id"],
                float(item["score"] or 0.0),
                item["preview"],
            )

        context = "\n\n---\n\n".join(context_parts)
        return analysis, context, context_parts, scores, source_chunks

    def _build_context_for_quality_retry(self, question: str, question_id: int = None):
        retry_query = self.rewrite_query(question)
        if "不存在控制关系" in (question or ""):
            retry_query = f"{retry_query} 不存在控制关系 关联方 企业"
        elif "未披露" in (question or "") and "关联方" in (question or ""):
            retry_query = f"{retry_query} 未披露 关联方"
        elif "关联方" in (question or ""):
            retry_query = f"{retry_query} 关联方 关联关系 控制关系"

        analysis = self.llm_engine.query_intent(question, question_id=question_id)
        source_file = self.select_pdf(question)
        analysis["source_file"] = source_file
        analysis["quality_retry_query"] = retry_query
        retrieved_results = self.vector_store.search(
            retry_query,
            max(Config.TOP_K_RETRIEVAL, 5),
            mode="bm25",
            source_file=source_file,
        )
        context_parts = []
        scores = []
        source_chunks = []
        for doc, score in retrieved_results:
            metadata = doc.metadata or {}
            context_parts.append(self._trim_context_part(doc.page_content))
            scores.append(score)
            source_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": metadata,
                    "relevance_score": score,
                    "auto_fallback": bool(metadata.get("auto_fallback")),
                    "fallback_from": metadata.get("fallback_from", ""),
                }
            )
        debug_preview = self._retrieval_debug_preview(source_chunks)
        auto_fallback_used = any(chunk.get("auto_fallback") for chunk in source_chunks)
        analysis["retrieved_count"] = len(source_chunks)
        analysis["retrieved_debug_preview"] = debug_preview
        analysis["auto_fallback"] = auto_fallback_used
        if auto_fallback_used:
            analysis["fallback_notice"] = "精确匹配无结果，已自动使用模糊匹配"
        for item in debug_preview:
            self.logger.info(
                "quality retry raw chunk | query=%s | source=%s | page=%s | chunk=%s | score=%.4f | preview=%s",
                retry_query,
                item["source_file"],
                item["page"],
                item["chunk_id"],
                float(item["score"] or 0.0),
                item["preview"],
            )
        return analysis, "\n\n---\n\n".join(context_parts), context_parts, scores, source_chunks
