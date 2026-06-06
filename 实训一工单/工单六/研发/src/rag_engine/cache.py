# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

from src.config import Config
from src.constants import NEGATIVE_QUERY_FALLBACK
from src.models import RAGResponse

class AnswerCacheMixin:
    def _answer_cache_key(
        self,
        question: str,
        question_id: int = None,
        retrieval_mode: str = "hybrid",
        session_id: str = "default",
        answer_language: str = "zh",
        retrieval_config: Dict | None = None,
    ) -> str:
        config_key = json.dumps(retrieval_config or {}, ensure_ascii=False, sort_keys=True)
        return "|".join(
            [
                question or "",
                str(question_id or ""),
                retrieval_mode or "",
                session_id or "",
                self._normalize_answer_language(answer_language),
                config_key,
            ]
        )

    def _get_cached_answer(self, cache_key: str, ttl_seconds: int = 30) -> RAGResponse | None:
        if not Config.ENABLE_ANSWER_CACHE:
            return None
        cached = self._answer_cache.get(cache_key)
        if not cached:
            return None
        response, timestamp = cached
        ttl = getattr(Config, "ANSWER_CACHE_TTL_SECONDS", ttl_seconds)
        if time.time() - timestamp <= ttl:
            return response
        self._answer_cache.pop(cache_key, None)
        return None

    def _cache_answer(self, cache_key: str, response: RAGResponse) -> RAGResponse:
        if not Config.ENABLE_ANSWER_CACHE:
            return response
        self._answer_cache[cache_key] = (response, time.time())
        return response

    def _cached_response(
        self,
        cache_key: str,
        *,
        question: str,
        answer: str,
        question_type: str,
        retrieval_mode: str,
        retrieved_contexts: List[str],
        scores: List[float],
        start_time: float,
        accuracy: float,
        source_chunks: List[Dict],
        has_context: bool,
        query_analysis: Dict,
        memory_hit: bool = False,
    ) -> RAGResponse:
        return self._cache_answer(
            cache_key,
            RAGResponse(
                question=question,
                answer=answer,
                question_type=question_type,
                memory_hit=memory_hit,
                retrieval_mode=retrieval_mode,
                retrieved_contexts=retrieved_contexts,
                scores=scores,
                response_time=time.time() - start_time,
                accuracy=accuracy,
                source_chunks=source_chunks,
                has_context=has_context,
                query_analysis=query_analysis,
            ),
        )

    def _negative_fallback_response(
        self,
        *,
        cache_key: str,
        question: str,
        question_type: str,
        answer_language: str,
        start_time: float,
        intent: str,
        context_label: str,
        reason: str,
    ) -> RAGResponse:
        localized_answer = self._localize_answer(question, NEGATIVE_QUERY_FALLBACK, answer_language)
        return self._cached_response(
            cache_key,
            question=question,
            answer=localized_answer,
            question_type=question_type,
            retrieval_mode="negative_query",
            retrieved_contexts=[context_label],
            scores=[0.0],
            start_time=start_time,
            accuracy=0.0,
            source_chunks=[
                {
                    "content": context_label,
                    "metadata": {"reason": reason},
                    "relevance_score": 0.0,
                }
            ],
            has_context=False,
            query_analysis={
                "intent": intent,
                "is_ambiguous": False,
                "answer_language": answer_language,
                "negative_query_results": [NEGATIVE_QUERY_FALLBACK],
            },
        )

    def _is_undisclosed_related_party_question(self, question: str) -> bool:
        q = question or ""
        if "不存在控制关系" in q:
            return False
        return "关联方" in q and any(marker in q for marker in ["未披露", "没有"])

    def _should_skip_long_term_memory(self, question: str, question_type: str) -> bool:
        """精确型招股书问题优先查 PDF，避免首次加载向量模型拖慢响应。"""
        if question_type in {"numeric", "percentage"}:
            return True
        exact_keywords = [
            "技术标准",
            "上游",
            "下游",
            "法定代表人",
            "注册资本",
            "募集资金",
            "重要供应商",
            "哪个领域",
            "国家科技进步一等奖",
            "一等奖",
            "荣获",
        ]
        return any(keyword in (question or "") for keyword in exact_keywords)
