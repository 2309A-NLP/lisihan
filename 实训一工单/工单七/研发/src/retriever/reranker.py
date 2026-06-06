# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""Reranking strategies for recalled RAG documents."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, List, Tuple

from src.document import Document


class Reranker:
    @staticmethod
    def _doc(item: Any) -> Document:
        return item[0] if isinstance(item, tuple) and item else item

    @staticmethod
    def _score(item: Any) -> float:
        if isinstance(item, tuple) and len(item) > 1:
            try:
                return float(item[1])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", (text or "").lower()):
            if re.fullmatch(r"[a-zA-Z0-9.]+", chunk):
                tokens.append(chunk)
            elif len(chunk) <= 2:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        return [token for token in tokens if token]

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, int, str]:
        metadata = doc.metadata or {}
        return (
            str(metadata.get("source_file", "")),
            int(metadata.get("page", 0) or 0),
            str(metadata.get("chunk_id", "")),
        )

    @staticmethod
    def llm_rerank(query: str, docs: list, top_k: int = 8) -> list:
        """使用LLM对召回结果重排。

        为保证离线可用和 3 秒响应约束，这里使用“LLM 风格”的语义特征代理：
        原召回分 + 查询词覆盖 + 短语命中奖励。外部 LLM 打分器可在此方法内替换。
        """
        query_text = query or ""
        query_terms = set(Reranker._tokenize(query_text))
        ranked = []
        for item in docs or []:
            doc = Reranker._doc(item)
            content = doc.page_content or ""
            doc_terms = set(Reranker._tokenize(content))
            overlap = len(query_terms & doc_terms)
            phrase_bonus = 2.0 if query_text and query_text in content else 0.0
            ranked.append((item, Reranker._score(item) + overlap + phrase_bonus))
        return [item for item, _ in sorted(ranked, key=lambda pair: pair[1], reverse=True)[:top_k]]

    @staticmethod
    def tfidf_rerank(query: str, docs: list, top_k: int = 8) -> list:
        """使用TF-IDF对召回结果重排。"""
        query_terms = Reranker._tokenize(query)
        if not query_terms:
            return list(docs or [])[:top_k]

        doc_terms = [Reranker._tokenize(Reranker._doc(item).page_content or "") for item in docs or []]
        doc_count = max(len(doc_terms), 1)
        df = Counter(term for terms in doc_terms for term in set(terms))
        query_counter = Counter(query_terms)
        ranked = []
        for item, terms in zip(docs or [], doc_terms):
            tf = Counter(terms)
            score = 0.0
            for term, query_freq in query_counter.items():
                if not tf.get(term):
                    continue
                idf = 1.0 + (doc_count / (1 + df.get(term, 0)))
                score += query_freq * tf[term] * idf
            ranked.append((item, score + Reranker._score(item)))
        return [item for item, _ in sorted(ranked, key=lambda pair: pair[1], reverse=True)[:top_k]]

    @staticmethod
    def adaptive_rerank(query: str, docs: list, user_feedback: dict, top_k: int = 8) -> list:
        """基于用户反馈的自适应重排。"""
        feedback = user_feedback or {}
        base = Reranker.tfidf_rerank(query, docs, top_k=len(docs or []))
        ranked = []
        for item in base:
            doc = Reranker._doc(item)
            key = Reranker._doc_key(doc)
            score = Reranker._score(item)
            score += float(feedback.get(key, feedback.get("|".join(map(str, key)), 0.0)) or 0.0)
            ranked.append((item, score))
        return [item for item, _ in sorted(ranked, key=lambda pair: pair[1], reverse=True)[:top_k]]
