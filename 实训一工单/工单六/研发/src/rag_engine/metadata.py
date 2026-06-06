# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import Dict, List

from src.processing.query_rewriter import _is_liyuan_document, rewrite_query, select_pdf
from src.processing.table_extractor import _extract_liyuan_table_answer
from utils.llm_engine import extract_complete_entity

class MetadataMixin:
    def document_metadata(self) -> Dict[str, str]:
        return self.metadata_cache.document_metadata

    def document_metadata(self, value: Dict[str, str]) -> None:
        self.metadata_cache.document_metadata = value

    def document_metadata_by_company(self) -> Dict[str, Dict[str, str]]:
        return self.metadata_cache.document_metadata_by_company

    def document_metadata_by_company(self, value: Dict[str, Dict[str, str]]) -> None:
        self.metadata_cache.document_metadata_by_company = value

    def _normalize_answer_language(self, answer_language: str = "zh") -> str:
        return self.llm_engine._normalize_answer_language(answer_language)

    def _localize_answer(self, question: str, answer: str, answer_language: str = "zh") -> str:
        return self.llm_engine.localize_answer(question, answer, answer_language)

    def _no_answer_for_low_quality(self, validation: Dict, answer_language: str = "zh") -> str:
        if self._normalize_answer_language(answer_language) == "en":
            return (
                "Based on the current knowledge base, I could not verify a reliable answer "
                f"for this question. Reason: {validation.get('reason', 'low_confidence')}."
            )
        return f"根据当前知识库，暂时无法验证出可靠答案。原因：{validation.get('reason', 'low_confidence')}。"

    def rewrite_query(self, question: str) -> str:
        return rewrite_query(question)

    def _is_liyuan_document(self, question: str) -> bool:
        return _is_liyuan_document(question)

    def select_pdf(self, question: str) -> str:
        return select_pdf(question)

    def _metadata_answer(self, question: str) -> str:
        return self.metadata_cache._metadata_answer(question)

    def _refresh_document_metadata(self) -> None:
        self.metadata_cache._refresh_document_metadata()

    def _metadata_source_chunk(self, question: str) -> Dict:
        return self.metadata_cache._metadata_source_chunk(question)

    def _estimate_accuracy(
        self,
        *,
        question: str,
        answer: str,
        retrieval_mode: str,
        has_context: bool,
        scores: List[float],
        retrieved_contexts: List[str],
        validation: Dict | None = None,
    ) -> float:
        if not answer or "暂时没有找到足够相关的信息" in answer:
            return 0.0
        if retrieval_mode in {"metadata", "table_metadata"}:
            return 1.0
        if retrieval_mode == "long_term_memory":
            return max(0.0, min(1.0, scores[0] if scores else 0.8))
        if not has_context:
            return 0.0
        if validation and validation.get("is_valid"):
            confidence = float(validation.get("confidence", 0.0))
            compact_answer = re.sub(r"\s+", "", answer or "")
            compact_context = re.sub(r"\s+", "", " ".join(retrieved_contexts or []))
            if compact_answer and (compact_answer[:80] in compact_context or any(term in compact_answer for term in ["技术标准", "技术规范"])):
                return max(0.95, confidence)
            return max(0.8, confidence)
        if re.search(r"\d", answer or "") and any(
            term in (question or "")
            for term in ["多少", "注册资本", "发行股数", "募集资金", "金额", "比例", "占比", "比重", "收入"]
        ):
            return 1.0

        keywords = self.llm_engine._extract_keywords(question)
        context = " ".join(retrieved_contexts)
        if keywords:
            relevance = sum(1 for keyword in keywords if keyword in context) / len(keywords)
        else:
            relevance = 0.5
        answer_keywords = self.llm_engine._extract_keywords(answer)
        if answer_keywords:
            faithfulness = sum(1 for keyword in answer_keywords if keyword in context) / len(answer_keywords)
        else:
            faithfulness = 0.5
        completeness = min(1.0, len(answer) / 20)
        return round((0.4 * relevance) + (0.4 * faithfulness) + (0.2 * completeness), 4)

    def _extract_liyuan_table_answer(self, question: str, context: str) -> str:
        return _extract_liyuan_table_answer(question, context)

    def _extract_complete_entity_answer(self, question: str, context: str) -> str:
        q = question or ""
        if "法定代表人" in q:
            complete = extract_complete_entity(context, "legal_representative")
            if complete:
                return complete.split("：", 1)[1]
        if "注册资本" in q:
            complete = extract_complete_entity(context, "registered_capital")
            if complete:
                return complete.split("：", 1)[1]
        return ""
