# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from src.config import Config, HYBRID_WEIGHTS, NUMERIC_KEYWORDS, PROPER_NOUN_KEYWORDS
from src.document import Document
from utils.logger import get_logger

from .cache import SearchCacheMixin
from .fusion import FusionMixin
from .models import _RetrievalHit, _StoredDocument
from .query import QueryProcessingMixin
from .reranker import Reranker
from .vector import VectorSearchMixin


logger = get_logger(__name__)


def get_dynamic_weights(question: str) -> dict:
    """根据问题类型动态返回BM25和向量检索的权重。"""
    question = question or ""
    for keyword in PROPER_NOUN_KEYWORDS:
        if keyword in question:
            return HYBRID_WEIGHTS["proper_noun"]
    for keyword in NUMERIC_KEYWORDS:
        if keyword in question:
            return HYBRID_WEIGHTS["numeric"]
    if len(question) > 20 and "什么" in question:
        return HYBRID_WEIGHTS["abstract"]
    return HYBRID_WEIGHTS["default"]


class HybridRetriever(QueryProcessingMixin, SearchCacheMixin, VectorSearchMixin, FusionMixin):
    """混合检索器：优先使用 BM25，必要时融合向量召回结果。"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or Config.COLLECTION_NAME
        retrieval_config = getattr(Config, "RETRIEVAL_CONFIG", {})
        hybrid_config = retrieval_config.get("hybrid", {})
        self.bm25_weight = float(hybrid_config.get("bm25_weight", getattr(Config, "BM25_RRF_WEIGHT", 3.0)))
        self.vector_weight = float(hybrid_config.get("vector_weight", getattr(Config, "VECTOR_RRF_WEIGHT", 1.0)))
        self.documents: List[_StoredDocument] = []
        self.bm25: Optional[BM25Okapi] = None
        self._embedding_model = None
        self._disabled_vector_backends = set()
        self._search_cache: OrderedDict[Tuple[Any, ...], List[Tuple[Document, float]]] = OrderedDict()
        self._search_cache_maxsize = 100
        self._user_feedback: Dict[Tuple[str, int, str], float] = {}
        self.last_weights = {"bm25": self.bm25_weight, "vector": self.vector_weight}
        self._query_synonyms = {
            "军用领域": ["国防领域", "军方客户", "军品业务", "军用"],
            "军用": ["国防领域", "军方客户", "军品业务", "直接和间接向国防客户", "来自军用领域"],
            "军用领域收入": ["直接和间接向国防客户的销售额合计", "销售额合计", "合计分别"],
            "民用领域": ["民用市场", "民品业务", "民用"],
            "民用": ["民用市场", "民品业务"],
            "收入": ["营业收入", "主营业务收入", "销售收入", "业务收入"],
            "比重": ["占主营业务收入的比重", "占比", "比例", "%"],
            "占比": ["占主营业务收入的比重", "比重", "比例", "%"],
            "上游": ["行业上下游情况", "电子元器件制造企业", "机箱", "机柜", "金属壳体制造企业"],
            "下游": ["行业上下游情况", "终端用户", "军队", "政府机关", "能源"],
            "电子信息行业": ["电子信息系统", "行业上下游情况", "行业竞争格局"],
            "技术标准": ["参与制定", "视频指挥系统技术标准", "某视频技术规范1.0", "全军第一个"],
            "国家科技进步一等奖": ["荣获国家科技进步一等奖", "某情报、指挥、控制与通信网络一体化工程"],
            "一等奖": ["国家科技进步一等奖", "某情报、指挥、控制与通信网络一体化工程"],
            "荣获": ["国家科技进步一等奖", "某情报、指挥、控制与通信网络一体化工程"],
            "募集资金拟投资": ["本次募集资金拟投资以下项目", "项目名称", "计划总投资"],
            "募集资金用途": ["本次募集资金拟投资以下项目", "项目名称", "计划总投资"],
            "重要供应商": ["国防军队视频指挥领域", "军队视频指挥领域", "视频指挥领域", "重要供应商"],
            "法定代表人": ["发行人的基本情况", "公司名称", "法定代表人"],
            "注册资本": ["发行人的基本情况", "公司名称", "注册资本"],
            "不存在控制关系": ["不存在控制关系的关联方", "企业名称", "与本公司关系"],
            "未披露": ["未披露", "不存在", "无"],
            "关联方": ["关联方及关联关系", "关联方名称", "企业名称", "与本公司关系"],
        }

    def create_vectorstore(self, documents: Sequence[Document]) -> None:
        if not documents:
            logger.warning("hybrid index skipped | reason=no_documents")
            self.documents = []
            self.bm25 = None
            return

        self.documents = [_StoredDocument(doc=doc, tokens=self._tokenize(doc.page_content)) for doc in documents]
        self.bm25 = BM25Okapi([item.tokens for item in self.documents])
        self._search_cache.clear()
        logger.info("hybrid bm25 index built | collection=%s | chunks=%s", self.collection_name, len(self.documents))

    def load_vectorstore(self):
        return self.collection_name if self.bm25 is not None and self.documents else None

    def _phrase_matches(self, query: str, content: str) -> bool:
        normalized_query = self._normalize_text(query)
        normalized_content = self._normalize_text(content)
        return bool(normalized_query and normalized_query in normalized_content)

    def _boolean_matches(self, query: str, content: str) -> bool:
        tokens = re.findall(r"AND|OR|NOT|[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", query or "", flags=re.I)
        if not tokens:
            return True
        content_tokens = set(self._tokenize(content))
        current: Optional[bool] = None
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
            term_tokens = self._tokenize(raw)
            matched = any(term in content_tokens or term in content for term in term_tokens)
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

    def _matches_bm25_type(self, query: str, content: str, match_type: str) -> bool:
        if match_type == "boolean":
            return self._boolean_matches(query, content)
        if match_type == "fuzzy":
            return self._fuzzy_matches(query, content)
        return self._phrase_matches(query, content) or self._has_query_overlap(query, content)

    def _bm25_search(self, query: str, top_k: int, source_file: str | None = None, match_type: str = "phrase") -> List[_RetrievalHit]:
        if not self.documents or self.bm25 is None:
            return []
        hits = self._bm25_search_once(query, top_k, source_file=source_file, match_type=match_type)
        if match_type != "fuzzy" and len(hits) < 2:
            fuzzy_hits = self._bm25_search_once(query, top_k, source_file=source_file, match_type="fuzzy")
            if len(fuzzy_hits) > len(hits):
                for hit in fuzzy_hits:
                    hit.auto_fallback = True
                    hit.doc = Document(
                        page_content=hit.doc.page_content,
                        metadata={**(hit.doc.metadata or {}), "auto_fallback": True, "fallback_from": match_type},
                    )
                logger.info(
                    "bm25 auto fallback to fuzzy | query=%s | source_file=%s | match_type=%s | original_hits=%s | fuzzy_hits=%s",
                    query,
                    source_file,
                    match_type,
                    len(hits),
                    len(fuzzy_hits),
                )
                return fuzzy_hits
        return hits

    def _bm25_search_once(self, query: str, top_k: int, source_file: str | None = None, match_type: str = "phrase") -> List[_RetrievalHit]:
        scores = self.bm25.get_scores(self._tokenize(self._expand_query(query)))
        candidate_limit = max(top_k * 8, int(getattr(Config, "BM25_FILTER_CANDIDATES", 80)))
        candidates = sorted(
            (
                (idx, score)
                for idx, score in enumerate(scores)
                if self._matches_source_file(self.documents[idx].doc, source_file)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_limit]
        ranked = sorted(
            (
                (idx, score)
                for idx, score in candidates
                if self._matches_bm25_type(query, self.documents[idx].doc.page_content, match_type)
            ),
            key=lambda item: item[1]
            + self._keyword_boost(query, self.documents[item[0]].doc.page_content)
            + (2.0 if self._phrase_matches(query, self.documents[item[0]].doc.page_content) else 0.0),
            reverse=True,
        )
        return [_RetrievalHit(self.documents[idx].doc, float(score)) for idx, score in ranked[:top_k]]

    def set_weights(self, bm25_weight: float, vector_weight: float):
        """动态调整 BM25 与向量检索权重。"""
        bm25_weight = max(0.0, float(bm25_weight))
        vector_weight = max(0.0, float(vector_weight))
        total = bm25_weight + vector_weight
        if total <= 0:
            bm25_weight = vector_weight = 0.5
            total = 1.0
        self.bm25_weight = bm25_weight / total
        self.vector_weight = vector_weight / total
        self._search_cache.clear()

    def _normalize_mode(self, mode: str) -> str:
        mode = (mode or "").lower()
        aliases = {
            "keyword": "bm25",
            "fulltext": "bm25",
            "full_text": "bm25",
            "rrf": "hybrid",
        }
        return aliases.get(mode, mode or "hybrid")

    def _search_options(self, retrieval_config: Dict | None = None) -> Dict:
        config = getattr(Config, "RETRIEVAL_CONFIG", {}).copy()
        if retrieval_config:
            config = {
                **config,
                **retrieval_config,
                "vector": {**config.get("vector", {}), **retrieval_config.get("vector", {})},
                "bm25": {**config.get("bm25", {}), **retrieval_config.get("bm25", {})},
                "hybrid": {**config.get("hybrid", {}), **retrieval_config.get("hybrid", {})},
            }
        return config

    def _filter_hits_by_source(self, hits: List[_RetrievalHit], source_file: str | None) -> List[_RetrievalHit]:
        return [hit for hit in hits if self._matches_source_file(hit.doc, source_file)]

    def _rerank_vector_hits(self, query: str, hits: List[_RetrievalHit], top_k: int, reranker: str) -> List[_RetrievalHit]:
        items = [(hit.doc, hit.score) for hit in hits]
        if reranker == "tfidf":
            reranked = Reranker.tfidf_rerank(query, items, top_k=top_k)
        elif reranker == "adaptive":
            reranked = Reranker.adaptive_rerank(query, items, self._user_feedback, top_k=top_k)
        else:
            reranked = Reranker.llm_rerank(query, items, top_k=top_k)
        return [_RetrievalHit(item[0], float(item[1])) if isinstance(item, tuple) else _RetrievalHit(item, 0.0) for item in reranked]

    def _vector_recall_search(self, query: str, top_k: int, source_file: str | None = None, reranker: str = "llm") -> List[_RetrievalHit]:
        hits = self._filter_hits_by_source(self._vector_search(query, max(top_k, getattr(Config, "VECTOR_K", top_k))), source_file)
        return self._rerank_vector_hits(query, hits, top_k, reranker)

    def record_feedback(self, source_chunks: List[Dict], helpful: bool) -> None:
        delta = 1.0 if helpful else -1.0
        for chunk in source_chunks or []:
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            key = (
                str(metadata.get("source_file", "")),
                int(metadata.get("page", 0) or 0),
                str(metadata.get("chunk_id", "")),
            )
            self._user_feedback[key] = self._user_feedback.get(key, 0.0) + delta

    def _min_max_normalize(self, hits: List[_RetrievalHit]) -> Dict[Tuple[Any, ...], float]:
        if not hits:
            return {}
        scores = [hit.score for hit in hits]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return {self._doc_key(hit.doc): 1.0 for hit in hits}
        return {self._doc_key(hit.doc): (hit.score - min_score) / (max_score - min_score) for hit in hits}

    def _weighted_average_fuse(
        self,
        bm25_hits: List[_RetrievalHit],
        vector_hits: List[_RetrievalHit],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        bm25_scores = self._min_max_normalize(bm25_hits)
        vector_scores = self._min_max_normalize(vector_hits)
        docs: Dict[Tuple[Any, ...], Document] = {}
        for hit in bm25_hits + vector_hits:
            docs.setdefault(self._doc_key(hit.doc), hit.doc)

        fused = []
        for key, doc in docs.items():
            score = self.bm25_weight * bm25_scores.get(key, 0.0) + self.vector_weight * vector_scores.get(key, 0.0)
            fused.append((doc, score))
        return sorted(fused, key=lambda item: item[1], reverse=True)[:top_k]

    def search(
        self,
        query: str,
        top_k: int = None,
        bm25_weight: float = None,
        vector_weight: float = None,
        mode: str = "hybrid",
        source_file: str | None = None,
        fusion_method: str = None,
        reranker: str = None,
        match_type: str = None,
        retrieval_config: Dict | None = None,
    ) -> List[Tuple[Document, float]]:
        final_top_k = top_k or Config.TOP_K_RETRIEVAL
        options = self._search_options(retrieval_config)
        requested_mode = (mode or options.get("mode", "hybrid")).lower()
        mode = self._normalize_mode(requested_mode)
        vector_options = options.get("vector", {})
        bm25_options = options.get("bm25", {})
        hybrid_options = options.get("hybrid", {})
        bm25_top_k = int(getattr(Config, "BM25_K", 10))
        vector_top_k = int(getattr(Config, "VECTOR_K", 10))
        reranker = reranker or vector_options.get("reranker", "llm")
        match_type = match_type or bm25_options.get("match_type", "phrase")
        fusion_method = fusion_method or ("rrf" if requested_mode == "rrf" else hybrid_options.get("fusion", "rrf"))
        if bm25_weight is None or vector_weight is None:
            if retrieval_config and "hybrid" in retrieval_config and (
                "bm25_weight" in retrieval_config["hybrid"] or "vector_weight" in retrieval_config["hybrid"]
            ):
                bm25_weight = retrieval_config["hybrid"].get("bm25_weight", self.bm25_weight)
                vector_weight = retrieval_config["hybrid"].get("vector_weight", self.vector_weight)
            else:
                weights = get_dynamic_weights(query)
                bm25_weight = weights["bm25"]
                vector_weight = weights["vector"]
        self.set_weights(bm25_weight, vector_weight)
        self.last_weights = {"bm25": self.bm25_weight, "vector": self.vector_weight}
        cache_key = (
            query or "",
            final_top_k,
            mode,
            source_file or "",
            fusion_method,
            reranker,
            match_type,
            round(self.bm25_weight, 4),
            round(self.vector_weight, 4),
        )
        cached = self._get_cached_search(cache_key)
        if cached is not None:
            return cached

        if mode == "bm25":
            results = [(hit.doc, hit.score) for hit in self._bm25_search(query, final_top_k, source_file=source_file, match_type=match_type)]
            self._set_cached_search(cache_key, results)
            return results
        if mode == "vector":
            results = [
                (hit.doc, hit.score)
                for hit in self._vector_recall_search(
                    query,
                    final_top_k,
                    source_file=source_file,
                    reranker=reranker,
                )
            ]
            self._set_cached_search(cache_key, results)
            return results

        bm25_hits = self._bm25_search(
            query,
            bm25_top_k,
            source_file=source_file,
            match_type=match_type,
        )
        if getattr(Config, "SKIP_VECTOR_FOR_EXACT_QUERIES", True) and self._is_exact_query(query):
            results = [(hit.doc, hit.score) for hit in bm25_hits[:final_top_k]]
            self._set_cached_search(cache_key, results)
            logger.info(
                "hybrid fast path used | query=%s | source_file=%s | bm25_hits=%s | fused_hits=%s",
                query,
                source_file,
                len(bm25_hits),
                len(results),
            )
            return results

        vector_hits = [
            hit
            for hit in self._vector_recall_search(
                query,
                vector_top_k,
                source_file=source_file,
                reranker=reranker,
            )
            if self._has_query_overlap(query, hit.doc.page_content)
        ]
        if fusion_method == "rrf":
            fused = self._rrf_fuse([bm25_hits, vector_hits], final_top_k)
        else:
            fused = self._weighted_average_fuse(bm25_hits, vector_hits, final_top_k)
        logger.info(
            "hybrid search done | query=%s | source_file=%s | bm25_hits=%s | vector_hits=%s | fused_hits=%s",
            query,
            source_file,
            len(bm25_hits),
            len(vector_hits),
            len(fused),
        )
        self._set_cached_search(cache_key, fused)
        return fused

    def search_with_relevance(self, query: str, top_k: int = None, mode: str = "hybrid", source_file: str | None = None, **kwargs) -> List[Dict]:
        return [
            {"content": doc.page_content, "score": score, "metadata": doc.metadata}
            for doc, score in self.search(query, top_k=top_k, mode=mode, source_file=source_file, **kwargs)
        ]

    def delete_collection(self):
        self.documents = []
        self.bm25 = None
        logger.info("hybrid retriever cleared | collection=%s", self.collection_name)

    def get_collection_stats(self) -> Dict:
        return {
            "exists": bool(self.documents),
            "name": self.collection_name,
            "count": len(self.documents),
            "backend": "hybrid",
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "bm25_top_k": getattr(Config, "BM25_K", 10),
            "vector_top_k": getattr(Config, "VECTOR_K", 10),
            "rrf_k": getattr(Config, "RRF_K", 60),
        }

    def list_vectors(self, limit: int = 20) -> List[Dict]:
        rows = []
        for item in self.documents[:limit]:
            metadata = item.doc.metadata or {}
            rows.append(
                {
                    "content": item.doc.page_content,
                    "source_file": metadata.get("source_file", ""),
                    "page": metadata.get("page", 0),
                    "chunk_id": metadata.get("chunk_id", ""),
                }
            )
        return rows
