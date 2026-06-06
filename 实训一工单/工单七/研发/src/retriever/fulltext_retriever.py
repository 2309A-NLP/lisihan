# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""Standalone BM25/full-text retriever with automatic fuzzy fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List

from rank_bm25 import BM25Okapi

from src.document import Document


@dataclass
class FulltextResult:
    document: Document
    score: float
    auto_fallback: bool = False


class FulltextRetriever:
    def __init__(self):
        self.documents: List[Document] = []
        self.tokens_by_doc: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip().lower()

    def _tokenize(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        tokens: List[str] = []
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", normalized):
            if re.fullmatch(r"[a-zA-Z0-9.]+", chunk):
                tokens.append(chunk.lower())
            elif len(chunk) <= 2:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        return [token for token in tokens if token]

    def build_index(self, documents: List[Document]):
        self.documents = list(documents or [])
        self.tokens_by_doc = [self._tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokens_by_doc) if self.tokens_by_doc else None

    def _phrase_matches(self, query: str, content: str) -> bool:
        return bool(self._normalize_text(query) and self._normalize_text(query) in self._normalize_text(content))

    def _boolean_matches(self, query: str, content: str) -> bool:
        tokens = re.findall(r"AND|OR|NOT|[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", query or "", flags=re.I)
        if not tokens:
            return True
        content_tokens = set(self._tokenize(content))
        current = None
        operator = "AND"
        negate_next = False
        for raw in tokens:
            token = raw.upper()
            if token in {"AND", "OR"}:
                operator = token
                continue
            if token == "NOT":
                negate_next = True
                continue
            matched = any(term in content_tokens or term in content for term in self._tokenize(raw))
            if negate_next:
                matched = not matched
                negate_next = False
            if current is None:
                current = matched
            elif operator == "OR":
                current = current or matched
            else:
                current = current and matched
            operator = "AND"
        return bool(current)

    def _fuzzy_matches(self, query: str, content: str) -> bool:
        query_terms = self._tokenize(query)
        content_terms = set(self._tokenize(content))
        if not query_terms:
            return True
        for term in query_terms:
            if term in content_terms or term in content:
                return True
            if any(
                abs(len(term) - len(candidate)) <= 2 and SequenceMatcher(None, term, candidate).ratio() >= 0.72
                for candidate in content_terms
            ):
                return True
        return False

    def _matches(self, query: str, content: str, match_type: str) -> bool:
        if match_type == "boolean":
            return self._boolean_matches(query, content)
        if match_type == "fuzzy":
            return self._fuzzy_matches(query, content)
        return self._phrase_matches(query, content)

    def _search_once(self, query: str, top_k: int, match_type: str) -> List[FulltextResult]:
        if not self.documents or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            (
                (idx, score)
                for idx, score in enumerate(scores)
                if self._matches(query, self.documents[idx].page_content, match_type)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [FulltextResult(self.documents[idx], float(score), auto_fallback=False) for idx, score in ranked[:top_k]]

    def search(self, query: str, top_k: int = 8, match_type: str = "phrase") -> List[Dict]:
        results = self._search_once(query, top_k=top_k, match_type=match_type)
        if match_type != "fuzzy" and len(results) < 2:
            fuzzy_results = self._search_once(query, top_k=top_k, match_type="fuzzy")
            if len(fuzzy_results) > len(results):
                for item in fuzzy_results:
                    item.auto_fallback = True
                results = fuzzy_results

        return [
            {
                "document": item.document,
                "content": item.document.page_content,
                "metadata": item.document.metadata,
                "score": item.score,
                "auto_fallback": item.auto_fallback,
            }
            for item in results
        ]
