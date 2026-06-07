# -*- coding: utf-8 -*-
"""Knowledge graph storage with optional Neo4j synchronization.

人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from src.document import Document
from utils.logger import get_logger

from .extractor import ExtractedEntity, ExtractedRelation, GraphEntityExtractor


@dataclass(frozen=True)
class GraphRelation:
    source: str
    relation: str
    target: str
    confidence: float
    evidence: str
    source_file: str
    page: int
    chunk_id: str


@dataclass(frozen=True)
class GraphStats:
    node_count: int
    relation_count: int
    chunk_count: int
    backend: str
    neo4j_available: bool


class KnowledgeGraphStore:
    """Stores document entities and relations for Graph RAG.

    The in-memory graph is always built. If Neo4j settings are present and the
    neo4j driver is installed, relations are also synchronized to Neo4j.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        enabled: bool | None = None,
    ):
        self.logger = get_logger(__name__)
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.enabled = os.getenv("ENABLE_NEO4J_GRAPH", "true").lower() == "true" if enabled is None else enabled
        self.extractor = GraphEntityExtractor()
        self.entities: Dict[str, ExtractedEntity] = {}
        self.relations: List[GraphRelation] = []
        self.chunk_lookup: Dict[str, Document] = {}
        self.entity_to_relations: dict[str, list[GraphRelation]] = defaultdict(list)
        self._driver = None
        self.neo4j_available = False
        self.last_error = ""
        self._connect()

    def _connect(self) -> None:
        if not self.enabled:
            self.last_error = "Neo4j disabled; using in-memory graph."
            return
        try:
            from neo4j import GraphDatabase

            if self.password:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            else:
                self._driver = GraphDatabase.driver(self.uri)
            self._driver.verify_connectivity()
            self.neo4j_available = True
            self.last_error = ""
            self.logger.info("neo4j connected | uri=%s | database=%s", self.uri, self.database)
        except Exception as exc:
            self._driver = None
            self.neo4j_available = False
            self.last_error = str(exc)
            self.logger.warning("neo4j unavailable; fallback to in-memory graph | error=%s", exc)

    @property
    def backend(self) -> str:
        return "neo4j+memory" if self.neo4j_available else "memory"

    def build_from_documents(self, documents: Iterable[Document]) -> GraphStats:
        docs = list(documents)
        self.entities.clear()
        self.relations.clear()
        self.chunk_lookup.clear()
        self.entity_to_relations.clear()

        for doc in docs:
            metadata = doc.metadata or {}
            key = self._chunk_key(
                metadata.get("source_file", ""),
                metadata.get("page", 0),
                metadata.get("chunk_id", ""),
            )
            self.chunk_lookup[key] = doc

        entities, relations = self.extractor.extract(docs)
        for entity in entities:
            self.entities[self._entity_key(entity.name)] = entity
        for item in relations:
            relation = GraphRelation(**asdict(item))
            self.relations.append(relation)
            self.entity_to_relations[self._entity_key(relation.source)].append(relation)
            self.entity_to_relations[self._entity_key(relation.target)].append(relation)

        if self.neo4j_available:
            self._sync_to_neo4j()

        stats = self.stats()
        self.logger.info(
            "knowledge graph built | backend=%s | nodes=%s | relations=%s | chunks=%s",
            stats.backend,
            stats.node_count,
            stats.relation_count,
            stats.chunk_count,
        )
        return stats

    def stats(self) -> GraphStats:
        nodes = set(self.entities.keys())
        for relation in self.relations:
            nodes.add(self._entity_key(relation.source))
            nodes.add(self._entity_key(relation.target))
        return GraphStats(
            node_count=len(nodes),
            relation_count=len(self.relations),
            chunk_count=len(self.chunk_lookup),
            backend=self.backend,
            neo4j_available=self.neo4j_available,
        )

    def find_entities(self, query: str, limit: int = 8) -> list[str]:
        query = query or ""
        if not query:
            return []
        normalized_query = self._normalize(query)
        scored: list[tuple[str, int]] = []
        for entity in self.entities.values():
            names = [entity.name, *entity.aliases]
            score = 0
            for name in names:
                normalized_name = self._normalize(name)
                if not normalized_name:
                    continue
                if normalized_name in normalized_query:
                    score = max(score, len(normalized_name) + 10)
                elif normalized_query in normalized_name and len(normalized_name) <= len(normalized_query) + 8:
                    score = max(score, len(normalized_query))
                elif self._token_overlap(normalized_query, normalized_name):
                    score = max(score, 3)
            if entity.type == "company" and len(entity.name) > 20 and not self._normalize(entity.name) in normalized_query:
                score = max(0, score - 6)
            if score:
                scored.append((entity.name, score))

        if not scored:
            for token in self._query_terms(query):
                key = self._entity_key(token)
                for relation in self.relations:
                    if key and (key in self._entity_key(relation.source) or key in self._entity_key(relation.target)):
                        scored.append((relation.source, 2))
                        scored.append((relation.target, 2))

        ordered = []
        seen = set()
        for name, _ in sorted(scored, key=lambda item: item[1], reverse=True):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
            if len(ordered) >= limit:
                break
        return ordered

    def related_relations(self, entities: Iterable[str], limit: int = 20) -> list[GraphRelation]:
        found: list[GraphRelation] = []
        seen = set()
        for entity in entities:
            for relation in self.entity_to_relations.get(self._entity_key(entity), []):
                key = self._relation_key(relation)
                if key not in seen:
                    found.append(relation)
                    seen.add(key)
                if len(found) >= limit:
                    return found
        return found

    def search_relations(self, query: str, limit: int = 20) -> list[GraphRelation]:
        if self.neo4j_available:
            neo4j_relations = self._search_neo4j(query, limit)
            if neo4j_relations:
                return neo4j_relations

        entities = self.find_entities(query, limit=8)
        related = self.related_relations(entities, limit=limit)
        if related:
            return self._rank_relations(query, related)[:limit]

        terms = self._query_terms(query)
        scored = []
        for relation in self.relations:
            haystack = self._normalize(" ".join([relation.source, relation.relation, relation.target, relation.evidence]))
            score = sum(1 for term in terms if self._normalize(term) in haystack)
            if score:
                scored.append((relation, score))
        return [item[0] for item in sorted(scored, key=lambda item: item[1], reverse=True)[:limit]]

    def _rank_relations(self, query: str, relations: list[GraphRelation]) -> list[GraphRelation]:
        return sorted(relations, key=lambda relation: self._relation_score(query, relation), reverse=True)

    def _relation_score(self, query: str, relation: GraphRelation) -> float:
        query = query or ""
        haystack = " ".join([relation.source, relation.relation, relation.target, relation.evidence])
        score = float(relation.confidence)
        for term in self._query_terms(query):
            if term in relation.relation:
                score += 5.0
            if term in relation.target:
                score += 3.0
            if term in relation.source:
                score += 2.0
            if term in haystack:
                score += 1.0
        priority_terms = {
            "法定代表人": ["法定代表人", "关联人物", "管理人员"],
            "董事长": ["董事长", "管理人员", "关联人物"],
            "营业收入": ["营业收入", "主营业务收入"],
            "净利润": ["净利润", "归属于母公司股东的净利润"],
            "信用减值损失": ["信用减值损失", "同比增长"],
            "组织架构": ["调整组织架构", "涉及组织"],
            "哑铃型": ["采用策略", "哑铃型"],
        }
        for trigger, preferred in priority_terms.items():
            if trigger in query and any(item in relation.relation or item in relation.target for item in preferred):
                score += 6.0
        return score

    def chunk_for_relation(self, relation: GraphRelation) -> Optional[Document]:
        key = self._chunk_key(relation.source_file, relation.page, relation.chunk_id)
        return self.chunk_lookup.get(key)

    def visualization_data(self, limit: int = 80, focus_entities: Iterable[str] | None = None) -> dict:
        focus = {self._entity_key(item) for item in (focus_entities or []) if item}
        relations = self.relations
        if focus:
            relations = [
                relation
                for relation in relations
                if self._entity_key(relation.source) in focus or self._entity_key(relation.target) in focus
            ]
        relations = relations[:limit]
        nodes = {}
        edges = []
        for relation in relations:
            for name in (relation.source, relation.target):
                key = self._entity_key(name)
                entity = self.entities.get(key)
                nodes[key] = {"id": key, "label": name, "type": entity.type if entity else "entity"}
            edges.append(
                {
                    "source": self._entity_key(relation.source),
                    "target": self._entity_key(relation.target),
                    "label": relation.relation,
                    "source_file": relation.source_file,
                    "page": relation.page,
                    "chunk_id": relation.chunk_id,
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}

    def visualization_data_for_query(self, query: str, limit: int = 40) -> dict:
        relations = self.search_relations(query, limit=limit)
        return self._relations_to_visualization(relations)

    def graphviz_dot(self, limit: int = 50, focus_entities: Iterable[str] | None = None) -> str:
        data = self.visualization_data(limit=limit, focus_entities=focus_entities)
        return self._graphviz_from_data(data)

    def graphviz_dot_for_query(self, query: str, limit: int = 40) -> str:
        data = self.visualization_data_for_query(query, limit=limit)
        return self._graphviz_from_data(data)

    def _relations_to_visualization(self, relations: Iterable[GraphRelation]) -> dict:
        nodes = {}
        edges = []
        for relation in relations:
            for name in (relation.source, relation.target):
                key = self._entity_key(name)
                entity = self.entities.get(key)
                node_type = entity.type if entity else self._infer_node_type(name)
                nodes[key] = {"id": key, "label": name, "type": node_type}
            edges.append(
                {
                    "source": self._entity_key(relation.source),
                    "target": self._entity_key(relation.target),
                    "label": relation.relation,
                    "source_file": relation.source_file,
                    "page": relation.page,
                    "chunk_id": relation.chunk_id,
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}

    def _graphviz_from_data(self, data: dict) -> str:
        lines = [
            "digraph G {",
            '  graph [rankdir=LR, bgcolor="transparent"];',
            '  node [shape=box, style="rounded,filled", fillcolor="#F7FAFC", color="#CBD5E1", fontname="Microsoft YaHei"];',
            '  edge [color="#64748B", fontname="Microsoft YaHei"];',
        ]
        for node in data["nodes"]:
            node_id = self._dot_id(node["id"])
            label = self._escape_dot(node["label"])
            fill = {
                "company": "#E0F2FE",
                "person": "#FCE7F3",
                "metric": "#DCFCE7",
                "strategy": "#FEF3C7",
                "organization": "#EDE9FE",
            }.get(node.get("type"), "#F7FAFC")
            lines.append(f'  {node_id} [label="{label}", fillcolor="{fill}"];')
        for edge in data["edges"]:
            label = self._escape_dot(edge["label"])
            lines.append(f'  {self._dot_id(edge["source"])} -> {self._dot_id(edge["target"])} [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)

    def _sync_to_neo4j(self) -> None:
        if not self._driver:
            return
        try:
            with self._driver.session(database=self.database) as session:
                session.run("CREATE CONSTRAINT graph_entity_name IF NOT EXISTS FOR (e:GraphEntity) REQUIRE e.name IS UNIQUE")
                session.run("MATCH (e:GraphEntity) DETACH DELETE e")
                for entity in self.entities.values():
                    label = self._neo4j_label(entity.type)
                    session.run(
                        f"""
                        MERGE (e:GraphEntity:{label} {{name: $name}})
                        SET e.type = $type, e.aliases = $aliases
                        """,
                        name=entity.name,
                        type=entity.type,
                        aliases=list(entity.aliases),
                    )
                for relation in self.relations:
                    source_label = self._neo4j_label(self._infer_node_type(relation.source))
                    target_type = self._infer_node_type(relation.target)
                    target_label = self._neo4j_label(target_type)
                    rel_type = self._neo4j_relation_type(relation.relation)
                    session.run(
                        f"""
                        MERGE (s:GraphEntity:{source_label} {{name: $source}})
                        MERGE (t:GraphEntity:{target_label} {{name: $target}})
                        SET t.type = coalesce(t.type, $target_type)
                        MERGE (s)-[r:{rel_type} {{relation: $relation, source_file: $source_file, page: $page, chunk_id: $chunk_id}}]->(t)
                        SET r.confidence = $confidence, r.evidence = $evidence
                        """,
                        source=relation.source,
                        target=relation.target,
                        target_type=target_type,
                        relation=relation.relation,
                        source_file=relation.source_file,
                        page=relation.page,
                        chunk_id=relation.chunk_id,
                        confidence=relation.confidence,
                        evidence=relation.evidence,
                    )
        except Exception as exc:
            self.logger.warning("neo4j typed sync failed; retrying with generic RELATED graph | error=%s", exc)
            self._sync_to_neo4j_plain()

    def _sync_to_neo4j_plain(self) -> None:
        if not self._driver:
            return
        try:
            with self._driver.session(database=self.database) as session:
                session.run("CREATE CONSTRAINT graph_entity_name IF NOT EXISTS FOR (e:GraphEntity) REQUIRE e.name IS UNIQUE")
                session.run("MATCH (e:GraphEntity) DETACH DELETE e")
                for entity in self.entities.values():
                    session.run(
                        """
                        MERGE (e:GraphEntity {name: $name})
                        SET e.type = $type, e.aliases = $aliases
                        """,
                        name=entity.name,
                        type=entity.type,
                        aliases=list(entity.aliases),
                    )
                for relation in self.relations:
                    session.run(
                        """
                        MERGE (s:GraphEntity {name: $source})
                        MERGE (t:GraphEntity {name: $target})
                        SET t.type = coalesce(t.type, $target_type)
                        MERGE (s)-[r:RELATED {relation: $relation, source_file: $source_file, page: $page, chunk_id: $chunk_id}]->(t)
                        SET r.confidence = $confidence, r.evidence = $evidence
                        """,
                        source=relation.source,
                        target=relation.target,
                        target_type=self._infer_node_type(relation.target),
                        relation=relation.relation,
                        source_file=relation.source_file,
                        page=relation.page,
                        chunk_id=relation.chunk_id,
                        confidence=relation.confidence,
                        evidence=relation.evidence,
                    )
        except Exception as exc:
            self.neo4j_available = False
            self.last_error = str(exc)
            self.logger.warning("neo4j sync failed; keep in-memory graph | error=%s", exc)

    def _search_neo4j(self, query: str, limit: int) -> list[GraphRelation]:
        if not self._driver:
            return []
        terms = self._query_terms(query)
        if not terms:
            return []
        try:
            with self._driver.session(database=self.database) as session:
                records = session.run(
                    """
                    MATCH (s:GraphEntity)-[r]->(t:GraphEntity)
                    WHERE any(term IN $terms WHERE s.name CONTAINS term OR t.name CONTAINS term OR r.relation CONTAINS term)
                    RETURN s.name AS source, r.relation AS relation, t.name AS target,
                           r.confidence AS confidence, r.evidence AS evidence,
                           r.source_file AS source_file, r.page AS page, r.chunk_id AS chunk_id
                    LIMIT $limit
                    """,
                    terms=terms,
                    limit=limit,
                )
                return [
                    GraphRelation(
                        source=record["source"],
                        relation=record["relation"],
                        target=record["target"],
                        confidence=float(record["confidence"] or 0.0),
                        evidence=record["evidence"] or "",
                        source_file=record["source_file"] or "",
                        page=int(record["page"] or 0),
                        chunk_id=record["chunk_id"] or "",
                    )
                    for record in records
                ]
        except Exception as exc:
            self.logger.warning("neo4j graph search failed | error=%s", exc)
            return []

    def _query_terms(self, query: str) -> list[str]:
        query = query or ""
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9.]+", query)
        domain_terms = [
            "技术标准",
            "重要供应商",
            "上游",
            "下游",
            "发行股数",
            "总股本",
            "募集资金",
            "投资项目",
            "控制关系",
            "关联方",
            "持股比例",
            "本公司关系",
            "法定代表人",
            "国家科技进步一等奖",
        ]
        for term in domain_terms:
            if term in query:
                terms.append(term)
        for token in re.findall(r"[\u4e00-\u9fff]{2}", query):
            if token in {"上游", "下游"}:
                terms.append(token)
        stopwords = {"什么", "哪些", "多少", "如何", "分别", "公司", "股份", "有限", "报告"}
        ordered = []
        seen = set()
        for term in terms:
            if term in stopwords or term in seen:
                continue
            ordered.append(term)
            seen.add(term)
        return ordered[:16]

    def _token_overlap(self, query: str, name: str) -> bool:
        if len(name) < 2:
            return False
        return any(name[idx : idx + 2] in query for idx in range(max(1, len(name) - 1)))

    def _chunk_key(self, source_file: object, page: object, chunk_id: object) -> str:
        return f"{source_file}|{int(page or 0)}|{chunk_id}"

    def _entity_key(self, name: str) -> str:
        return self._normalize(name)

    def _relation_key(self, relation: GraphRelation) -> tuple:
        return (
            relation.source,
            relation.relation,
            relation.target,
            relation.source_file,
            relation.page,
            relation.chunk_id,
        )

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "").lower()

    def _dot_id(self, value: str) -> str:
        return "n_" + re.sub(r"[^0-9A-Za-z_]", "_", value)

    def _escape_dot(self, value: str) -> str:
        return (value or "").replace("\\", "\\\\").replace('"', '\\"')[:80]

    def _infer_node_type(self, name: str) -> str:
        entity = self.entities.get(self._entity_key(name))
        if entity:
            return entity.type
        if re.fullmatch(r"[0-9,，.]+(?:万|亿)?(?:元|股)|[0-9.]+%", name or ""):
            return "value"
        return "entity"

    def _neo4j_label(self, entity_type: str) -> str:
        return {
            "company": "Company",
            "person": "Person",
            "metric": "Metric",
            "strategy": "Strategy",
            "organization": "Organization",
            "value": "Value",
        }.get(entity_type, "Entity")

    def _neo4j_relation_type(self, relation: str) -> str:
        if relation in {"法定代表人", "管理人员", "关联人物"}:
            return "`人员`"
        if relation in {"采用策略"}:
            return "`采用`"
        if relation in {"上游涉及", "下游包括"}:
            return "`产业链`"
        if relation in {"参与制定技术标准", "成为重要供应商领域", "荣获国家科技进步一等奖"}:
            return "`业务能力`"
        if relation in {"募集资金拟投资项目"}:
            return "`募投项目`"
        if relation in {"存在控制关系关联方", "不存在控制关系关联方", "持股比例", "本公司关系"}:
            return "`关联方`"
        if relation == "调整组织架构":
            return "`调整`"
        if relation == "涉及组织":
            return "`设有`"
        if "同比增长" in relation:
            return "`增长`"
        if relation in {
            "营业收入",
            "主营业务收入",
            "净利润",
            "归属于母公司股东的净利润",
            "信用减值损失",
            "注册资本",
            "募集资金",
            "发行股数",
            "发行后总股本",
            "占发行后总股本比例",
            "总股本",
        }:
            return "`指标`"
        return "`相关`"
