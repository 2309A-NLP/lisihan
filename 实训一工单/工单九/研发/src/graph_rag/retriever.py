# -*- coding: utf-8 -*-
"""Graph RAG retrieval helpers."""

from __future__ import annotations

import re
from typing import Dict, List

from src.document import Document

from .graph_store import GraphRelation, KnowledgeGraphStore


class GraphRAGRetriever:
    """Retrieves graph relations and related source chunks for a question."""

    def __init__(self, graph_store: KnowledgeGraphStore):
        self.graph_store = graph_store

    def search(self, question: str, top_k: int = 4, source_file: str | None = None) -> list[tuple[Document, float, Dict]]:
        relations = self.graph_store.search_relations(question, limit=max(top_k * 4, 12))
        if source_file:
            preferred = [item for item in relations if item.source_file == source_file]
            fallback = [item for item in relations if item.source_file != source_file]
            relations = preferred + fallback

        hits: list[tuple[Document, float, Dict]] = []
        seen_chunks = set()
        for rank, relation in enumerate(relations, start=1):
            chunk = self.graph_store.chunk_for_relation(relation)
            content = self._build_context_text(relation, chunk)
            key = (relation.source_file, relation.page, relation.chunk_id, relation.source, relation.relation, relation.target)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            doc = Document(
                page_content=content,
                metadata={
                    "source_file": relation.source_file,
                    "page": relation.page,
                    "chunk_id": relation.chunk_id,
                    "type": "graph_relation",
                    "graph_source": relation.source,
                    "graph_relation": relation.relation,
                    "graph_target": relation.target,
                    "graph_backend": self.graph_store.backend,
                },
            )
            score = max(0.1, relation.confidence) + max(0.0, (top_k * 4 - rank) / 100)
            hits.append((doc, score, {"relation": relation}))
            if len(hits) >= top_k:
                break
        return hits

    def explain(self, question: str, limit: int = 8) -> dict:
        entities = self.graph_store.find_entities(question, limit=limit)
        relations = self.graph_store.related_relations(entities, limit=limit) if entities else self.graph_store.search_relations(question, limit=limit)
        return {
            "matched_entities": entities,
            "relations": [self._relation_payload(item) for item in relations],
            "stats": self.graph_store.stats().__dict__,
            "backend": self.graph_store.backend,
            "neo4j_error": self.graph_store.last_error,
        }

    def _build_context_text(self, relation: GraphRelation, chunk: Document | None) -> str:
        pieces = [
            f"[知识图谱关系] {relation.source} --{relation.relation}--> {relation.target}",
            f"[来源] {relation.source_file} 第{relation.page}页 chunk={relation.chunk_id}",
        ]
        evidence = relation.evidence
        if chunk is not None and chunk.page_content:
            evidence = chunk.page_content
        evidence = re.sub(r"\s+", " ", evidence or "").strip()
        if evidence:
            pieces.append(f"[原文证据] {evidence[:1000]}")
        return "\n".join(pieces)

    def _relation_payload(self, relation: GraphRelation) -> dict:
        return {
            "source": relation.source,
            "relation": relation.relation,
            "target": relation.target,
            "confidence": relation.confidence,
            "source_file": relation.source_file,
            "page": relation.page,
            "chunk_id": relation.chunk_id,
        }
