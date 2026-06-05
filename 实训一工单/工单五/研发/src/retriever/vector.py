# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import Config
from src.document import Document
from utils.logger import get_logger

from .models import _RetrievalHit


logger = get_logger(__name__)


class VectorSearchMixin:
    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            return self._embedding_model
        except Exception:
            logger.exception("embedding model load failed | model=%s", Config.EMBEDDING_MODEL)
            return None

    def _embed_query(self, query: str) -> Optional[List[float]]:
        model = self._load_embedding_model()
        if model is None:
            return None
        vector = model.encode([query], normalize_embeddings=True)[0]
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def _entity_to_document(self, entity: Dict[str, Any]) -> Document:
        content = entity.get("content") or entity.get("page_content") or entity.get("text") or ""
        metadata = entity.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"metadata": metadata}

        for key in ("source_file", "page", "chunk_id", "source_path"):
            if key in entity and key not in metadata:
                metadata[key] = entity[key]
        return Document(page_content=content, metadata=metadata)

    def _milvus_output_fields(self, client: Any) -> List[str]:
        try:
            schema = client.describe_collection(self.collection_name)
            existing_fields = {field.get("name") for field in schema.get("fields", [])}
        except Exception as exc:
            logger.warning("milvus schema lookup failed | collection=%s | error=%s", self.collection_name, exc)
            existing_fields = set(Config.MILVUS_OUTPUT_FIELDS)

        requested = [field for field in Config.MILVUS_OUTPUT_FIELDS if field in existing_fields]
        for field in ("content", "source_file", "page", "chunk_id"):
            if field in existing_fields and field not in requested:
                requested.append(field)
        return requested

    def _milvus_search(self, query: str, top_k: int) -> List[_RetrievalHit]:
        if "milvus" in self._disabled_vector_backends:
            return []
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=Config.MILVUS_URI, timeout=Config.MILVUS_TIMEOUT)
            query_vector = self._embed_query(query)
            if query_vector is None:
                return []
            results = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field=Config.MILVUS_VECTOR_FIELD,
                limit=top_k,
                output_fields=self._milvus_output_fields(client),
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            )
        except Exception as exc:
            self._disabled_vector_backends.add("milvus")
            logger.warning("milvus vector search unavailable | collection=%s | error=%s", self.collection_name, exc)
            return []

        hits: List[_RetrievalHit] = []
        for hit in (results[0] if results else []):
            entity = hit.get("entity", hit) if isinstance(hit, dict) else getattr(hit, "entity", {})
            score = hit.get("distance", hit.get("score", 0.0)) if isinstance(hit, dict) else getattr(hit, "distance", 0.0)
            doc = self._entity_to_document(entity or {})
            if doc.page_content:
                hits.append(_RetrievalHit(doc, float(score)))
        return hits

    def _vector_search(self, query: str, top_k: int) -> List[_RetrievalHit]:
        return self._milvus_search(query, top_k)
