# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from src.config import Config
from src.document import Document
from utils.logger import get_logger

from .cache import SearchCacheMixin
from .fusion import FusionMixin
from .models import _RetrievalHit, _StoredDocument
from .query import QueryProcessingMixin
from .vector import VectorSearchMixin


logger = get_logger(__name__)


class HybridRetriever(QueryProcessingMixin, SearchCacheMixin, VectorSearchMixin, FusionMixin):
    """混合检索器：优先使用 BM25，必要时融合向量召回结果。"""

    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.documents: List[_StoredDocument] = []
        self.bm25: Optional[BM25Okapi] = None
        self._embedding_model = None
        self._disabled_vector_backends = set()
        self._search_cache: OrderedDict[Tuple[str, int, str, str], List[Tuple[Document, float]]] = OrderedDict()
        self._search_cache_maxsize = 100
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

    def _bm25_search(self, query: str, top_k: int, source_file: str | None = None) -> List[_RetrievalHit]:
        if not self.documents or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(self._tokenize(self._expand_query(query)))
        ranked = sorted(
            ((idx, score) for idx, score in enumerate(scores) if self._matches_source_file(self.documents[idx].doc, source_file)),
            key=lambda item: item[1] + self._keyword_boost(query, self.documents[item[0]].doc.page_content),
            reverse=True,
        )
        return [_RetrievalHit(self.documents[idx].doc, float(score)) for idx, score in ranked[:top_k]]

    def search(self, query: str, top_k: int = None, mode: str = "hybrid", source_file: str | None = None) -> List[Tuple[Document, float]]:
        final_top_k = top_k or Config.TOP_K_RETRIEVAL
        bm25_top_k = getattr(Config, "BM25_K", 10)
        vector_top_k = getattr(Config, "VECTOR_K", 10)
        cache_key = (query or "", final_top_k, mode, source_file or "")
        cached = self._get_cached_search(cache_key)
        if cached is not None:
            return cached

        if mode == "bm25":
            results = [(hit.doc, hit.score) for hit in self._bm25_search(query, final_top_k, source_file=source_file)]
            self._set_cached_search(cache_key, results)
            return results
        if mode == "vector":
            results = [
                (hit.doc, hit.score)
                for hit in self._vector_search(query, final_top_k)
                if self._matches_source_file(hit.doc, source_file)
            ]
            self._set_cached_search(cache_key, results)
            return results

        bm25_hits = self._bm25_search(query, bm25_top_k, source_file=source_file)
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
            for hit in self._vector_search(query, vector_top_k)
            if self._matches_source_file(hit.doc, source_file) and self._has_query_overlap(query, hit.doc.page_content)
        ]
        fused = self._rrf_fuse([bm25_hits, vector_hits], final_top_k)
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

    def search_with_relevance(self, query: str, top_k: int = None, mode: str = "hybrid", source_file: str | None = None) -> List[Dict]:
        return [
            {"content": doc.page_content, "score": score, "metadata": doc.metadata}
            for doc, score in self.search(query, top_k=top_k, mode=mode, source_file=source_file)
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
