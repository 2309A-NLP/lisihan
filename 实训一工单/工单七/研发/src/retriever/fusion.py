# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.config import Config
from src.document import Document
from utils.logger import get_logger

from .models import _RetrievalHit


logger = get_logger(__name__)


class FusionMixin:
    def _doc_key(self, doc: Document) -> Tuple[Any, ...]:
        metadata = doc.metadata or {}
        return (
            metadata.get("source_file"),
            metadata.get("page"),
            metadata.get("chunk_id"),
            doc.page_content[:120],
        )

    def _rrf_fuse(self, rankings: Sequence[Sequence[_RetrievalHit]], top_k: int) -> List[Tuple[Document, float]]:
        rrf_k = getattr(Config, "RRF_K", 60)
        retrieval_config = getattr(Config, "RETRIEVAL_CONFIG", {})
        hybrid_config = retrieval_config.get("hybrid", {})
        bm25_weight = getattr(self, "bm25_weight", hybrid_config.get("bm25_weight", getattr(Config, "BM25_RRF_WEIGHT", 2.0)))
        vector_weight = getattr(self, "vector_weight", hybrid_config.get("vector_weight", getattr(Config, "VECTOR_RRF_WEIGHT", 1.0)))
        fused_scores: Dict[Tuple[Any, ...], float] = {}
        documents: Dict[Tuple[Any, ...], Document] = {}

        for ranking_index, ranking in enumerate(rankings):
            weight = bm25_weight if ranking_index == 0 else vector_weight
            for rank, hit in enumerate(ranking, start=1):
                key = self._doc_key(hit.doc)
                documents.setdefault(key, hit.doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + weight / (rrf_k + rank)

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [(documents[key], score) for key, score in ranked[:top_k]]
