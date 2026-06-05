# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from src.config import Config

class ContextMixin:
    def _context_text_for_doc(self, doc) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        parent_content = metadata.get("parent_content")
        if metadata.get("chunk_strategy") in {"combined", "parent_child"} and parent_content:
            return str(parent_content)
        return getattr(doc, "page_content", "") or ""

    def _build_context(self, question: str, top_k: int = 4, question_id: int = None, retrieval_mode: str = "hybrid"):
        analysis = self.llm_engine.query_intent(question, question_id=question_id)
        search_query = self.rewrite_query(question)
        source_file = self.select_pdf(question)
        if search_query != question:
            analysis["rewritten_query"] = search_query
        analysis["source_file"] = source_file
        retrieved_results = self.vector_store.search(search_query, top_k, mode=retrieval_mode, source_file=source_file)

        context_parts = []
        scores = []
        source_chunks = []
        for doc, score in retrieved_results:
            context_text = self._context_text_for_doc(doc)
            context_parts.append(context_text)
            scores.append(score)
            source_chunks.append(
                {
                    "content": context_text,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
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
            context_text = self._context_text_for_doc(doc)
            context_parts.append(context_text)
            scores.append(score)
            source_chunks.append(
                {
                    "content": context_text,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
            )
        return analysis, "\n\n---\n\n".join(context_parts), context_parts, scores, source_chunks
