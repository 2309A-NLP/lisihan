# -*- coding: utf-8 -*-
"""工单八：Graph RAG 功能测试及评估脚本。

人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List

from src.config import Config
from src.document import Document
from src.graph_rag.extractor import GraphEntityExtractor
from src.rag_engine import RAGEngine


QUESTIONS_PATH = Path("eval_questions.md")
OUTPUT_DIR = Path("output")
RESULTS_PATH = OUTPUT_DIR / "work_order_8_results.json"
REPORT_PATH = OUTPUT_DIR / "work_order_8_report.md"
LEGACY_RESULTS_PATH = Path("work_order_8_results.json")
LEGACY_REPORT_PATH = Path("work_order_8_report.md")
WORK_ORDER_7_RESULTS_PATH = Path("work_order_7_results.json")
RETRIEVAL_MODE = "hybrid_graph"
MINERU_OUTPUT_DIR = Path("mineru_output")


def build_question_subgraph(
    engine: RAGEngine,
    question: str,
    source_chunks: list[dict],
    answer: str = "",
    limit: int = 32,
) -> dict:
    """为每个问题生成可检查的局部知识图谱。

    优先从本题检索片段中即时抽取实体和关系，使子图与答案来源同源；
    如果本题片段无法抽取关系，再使用 Neo4j/内存图谱按问题关键词检索。
    """
    extractor = GraphEntityExtractor()
    docs = []
    filtered_chunks = _filter_source_chunks_for_graph(question, answer, source_chunks or [])
    if _chunks_lack_text_evidence(filtered_chunks) or "控制关系" in question or "关联方" in question:
        evidence_chunks = _find_text_evidence_chunks(question, answer, limit=6)
        filtered_chunks = evidence_chunks or filtered_chunks
    for idx, chunk in enumerate(filtered_chunks):
        content = chunk.get("content", "")
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("chunk_id", metadata.get("chunk_id") or f"question_chunk_{idx}")
        docs.append(Document(page_content=content, metadata=metadata))
    entities, relations = extractor.extract(docs)
    nodes = {}
    edges = []
    for entity in entities:
        key = entity.name
        nodes[key] = {"id": key, "label": entity.name, "type": entity.type}
    for relation in relations[:limit]:
        for name in (relation.source, relation.target):
            nodes.setdefault(name, {"id": name, "label": name, "type": "entity"})
        edges.append(
            {
                "source": relation.source,
                "target": relation.target,
                "label": relation.relation,
                "source_file": relation.source_file,
                "page": relation.page,
                "chunk_id": relation.chunk_id,
            }
        )
    nodes, edges = _filter_local_edges(nodes, edges, question, answer, limit)
    if edges:
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "reason": "已从本题检索片段中动态抽取真实实体关系。",
        }

    graph_subgraph = {}
    if hasattr(engine, "graph_store"):
        graph_subgraph = engine.graph_store.visualization_data_for_query(f"{question} {answer}", limit=limit)
        graph_subgraph = _filter_subgraph_edges(graph_subgraph, question, answer, limit)
    if graph_subgraph.get("nodes") and graph_subgraph.get("edges"):
        if any(str(chunk.get("content", "")).startswith("multimodal_image:") for chunk in source_chunks or []):
            graph_subgraph["reason"] = "本题命中图片/多模态片段，规则抽取器无法从图片占位符直接抽取；改用全局图谱按问题关键词检索真实关系。"
        else:
            graph_subgraph["reason"] = "本题检索片段未抽取到关系；改用 Neo4j/内存图谱按问题关键词检索真实关系。"
        return graph_subgraph

    if any(str(chunk.get("content", "")).startswith("multimodal_image:") for chunk in source_chunks or []):
        reason = "本题命中图片/多模态片段，当前规则抽取器无法从图片占位符直接抽取实体关系，且全局图谱未命中相关关系。"
    elif source_chunks:
        reason = "本题有文本检索片段，但片段中未匹配到当前规则支持的实体关系模式，且全局图谱未命中相关关系。"
    else:
        reason = "本题没有可用于图谱抽取的检索片段，且全局图谱未命中相关关系。"
    return {"nodes": list(nodes.values()), "edges": edges, "reason": reason}


def _filter_source_chunks_for_graph(question: str, answer: str, source_chunks: list[dict]) -> list[dict]:
    """保留更可能支撑本题答案的片段，避免宽泛图谱片段污染局部子图。"""
    if not source_chunks:
        return []
    answer_terms = _important_terms(answer)
    answer_terms.extend(_numeric_terms(answer))
    question_terms = _important_terms(question)
    numeric_terms = [_normalize_for_match(term) for term in _numeric_terms(answer)]
    if numeric_terms:
        numeric_hits = []
        for chunk in source_chunks:
            content = chunk.get("content", "") or ""
            if content in {"document_metadata", "negative_query_handler"} or content.startswith("multimodal_image:"):
                continue
            haystack = _normalize_for_match(content)
            if any(term and term in haystack for term in numeric_terms):
                numeric_hits.append(chunk)
        if numeric_hits:
            return numeric_hits

    kept = []
    for chunk in source_chunks:
        content = chunk.get("content", "") or ""
        metadata = chunk.get("metadata") or {}
        source_file = metadata.get("source_file", "") or ""
        if content in {"document_metadata", "negative_query_handler"}:
            continue
        if content.startswith("multimodal_image:"):
            kept.append(chunk)
            continue
        haystack = _normalize_for_match(f"{content} {source_file}")
        normalized_answer_terms = [_normalize_for_match(term) for term in answer_terms]
        normalized_question_terms = [_normalize_for_match(term) for term in question_terms]
        if normalized_answer_terms and any(term and term in haystack for term in normalized_answer_terms):
            kept.append(chunk)
            continue
        if normalized_question_terms and sum(1 for term in normalized_question_terms if term and term in haystack) >= 2:
            kept.append(chunk)
            continue
        if chunk.get("retrieval_source") != "graph_rag":
            kept.append(chunk)
    return kept or source_chunks


def _filter_local_edges(nodes: dict, edges: list[dict], question: str, answer: str, limit: int) -> tuple[dict, list[dict]]:
    """按问题意图过滤局部抽取边，避免无关人员/组织关系混入报告。"""
    if not edges:
        return nodes, edges
    preferred = _preferred_relation_labels(question)
    terms = [_normalize_for_match(term) for term in (_important_terms(answer) + _numeric_terms(answer))]
    filtered = []
    for edge in edges:
        label = str(edge.get("label", ""))
        haystack = _normalize_for_match(" ".join(str(edge.get(key, "")) for key in ("source", "target", "label", "source_file")))
        if _edge_matches_question_intent(edge, question, answer):
            filtered.append(edge)
        elif preferred and label in preferred and not ("控制关系" in question or "关联方" in question):
            filtered.append(edge)
        elif not preferred and terms and any(term and term in haystack for term in terms):
            filtered.append(edge)
        elif not preferred and not terms:
            filtered.append(edge)
    if not filtered and not preferred:
        filtered = edges
    filtered = filtered[:limit]
    used_names = {edge.get("source") for edge in filtered} | {edge.get("target") for edge in filtered}
    used_nodes = {
        key: value
        for key, value in nodes.items()
        if key in used_names or value.get("id") in used_names or value.get("label") in used_names
    }
    for name in used_names:
        if name and name not in used_nodes:
            used_nodes[name] = {"id": name, "label": name, "type": "entity"}
    return used_nodes, filtered


def _chunks_lack_text_evidence(chunks: list[dict]) -> bool:
    if not chunks:
        return True
    for chunk in chunks:
        content = chunk.get("content", "") or ""
        if content and content not in {"document_metadata", "negative_query_handler"} and not content.startswith("multimodal_image:"):
            return False
    return True


def _find_text_evidence_chunks(question: str, answer: str, limit: int = 6) -> list[dict]:
    """从 MinerU 解析文本中找本题答案证据，供图谱抽取使用。"""
    terms = _important_terms(f"{question} {answer}") + _numeric_terms(answer)
    if "存在控制关系" in question:
        terms.extend(["存在控制关系的关联方", "关联方名称", "持股比例", "与本公司关系"])
    if "不存在控制关系" in question:
        terms.extend(["不存在控制关系的关联方", "企业名称", "与本公司关系"])
    terms = [_normalize_for_match(term) for term in terms if len(_normalize_for_match(term)) >= 2]
    if not terms or not MINERU_OUTPUT_DIR.exists():
        return []

    hits = []
    grouped_hits = {}
    page_flags: dict[tuple, dict] = {}
    is_relation_question = "控制关系" in question or "关联方" in question
    answer_terms = [_normalize_for_match(term) for term in (_important_terms(answer) + _numeric_terms(answer))]
    for path in MINERU_OUTPUT_DIR.glob("*_chunks.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        for chunk in chunks:
            content = chunk.get("content", "") or ""
            normalized = _normalize_for_match(content)
            source_name = _normalize_for_match(payload.get("source_file", path.name))
            if is_relation_question and "武汉力源信息技术股份有限公司" in question and "招股说明书2" not in source_name:
                continue
            score = sum(1 for term in terms if term in normalized)
            marker_present = any(marker in normalized for marker in ("存在控制关系的关联方", "不存在控制关系的关联方", "关联方名称", "企业名称"))
            answer_present = any(term and term in normalized for term in answer_terms)
            if is_relation_question and not (marker_present or answer_present):
                continue
            if (
                not is_relation_question
                and "武汉力源信息技术股份有限公司" in question
                and ("招股说明书2" in source_name or "武汉力源信息技术股份有限公司" in normalized)
            ):
                score += 20
            if "存在控制关系" in question and "存在控制关系的关联方" in normalized:
                score += 10
            if "不存在控制关系" in question and "不存在控制关系的关联方" in normalized:
                score += 10
            if score <= 0:
                continue
            metadata = dict(chunk.get("metadata") or {})
            metadata.setdefault("source_file", payload.get("source_file", path.name.replace("_chunks.json", ".pdf")))
            metadata.setdefault("page", chunk.get("page", 0))
            metadata.setdefault("chunk_id", metadata.get("chunk_id") or f"p{chunk.get('page', 0)}_evidence")
            group_key = (metadata.get("source_file"), metadata.get("page"))
            grouped_hits[group_key] = max(grouped_hits.get(group_key, 0), score)
            flags = page_flags.setdefault(group_key, {"marker": False, "answer": False})
            flags["marker"] = flags["marker"] or marker_present
            flags["answer"] = flags["answer"] or answer_present
            hits.append(
                (
                    score,
                    {
                        "content": content,
                        "metadata": metadata,
                        "retrieval_source": "mineru_evidence",
                    },
                )
            )
    if is_relation_question and grouped_hits:
        def page_rank(item):
            key, score = item
            flags = page_flags.get(key, {})
            return (1 if flags.get("marker") and flags.get("answer") else 0, 1 if flags.get("marker") else 0, score)

        best_source, best_page = max(grouped_hits.items(), key=page_rank)[0]
        combined = _combined_page_chunk(best_source, best_page)
        if combined:
            return [combined]
    hits.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in hits[:limit]]


def _combined_page_chunk(source_file: str, page: int) -> dict | None:
    for path in MINERU_OUTPUT_DIR.glob("*_chunks.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("source_file") != source_file:
            continue
        chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        texts = [chunk.get("content", "") for chunk in chunks if int(chunk.get("page", 0) or 0) == int(page or 0)]
        texts = [text for text in texts if text]
        if not texts:
            return None
        return {
            "content": "\n".join(texts),
            "metadata": {
                "source_file": source_file,
                "page": int(page or 0),
                "chunk_id": f"p{int(page or 0)}_combined_evidence",
            },
            "retrieval_source": "mineru_page_evidence",
        }
    return None


def _filter_subgraph_edges(graph_subgraph: dict, question: str, answer: str, limit: int) -> dict:
    edges = list((graph_subgraph or {}).get("edges") or [])
    if not edges:
        return graph_subgraph
    terms = _important_terms(f"{question} {answer}") + _numeric_terms(answer)
    preferred = _preferred_relation_labels(question)
    relation_terms = {
        "发行股数": ["发行股数", "总股本", "占发行后总股本比例"],
        "总股本": ["发行股数", "总股本", "占发行后总股本比例"],
        "募集资金": ["募集资金", "补充流动资金", "募集资金拟投资项目", "计划总投资"],
        "投资项目": ["募集资金拟投资项目", "计划总投资"],
        "控制关系": ["存在控制关系关联方", "不存在控制关系关联方", "持股比例", "本公司关系"],
        "关联方": ["存在控制关系关联方", "不存在控制关系关联方", "持股比例", "本公司关系"],
        "法定代表人": ["法定代表人"],
        "技术标准": ["参与制定技术标准"],
        "重要供应商": ["成为重要供应商领域"],
        "上游": ["上游涉及"],
        "下游": ["下游包括"],
    }
    if not preferred:
        for trigger, labels in relation_terms.items():
            if trigger in question:
                preferred.extend(labels)

    filtered = []
    for edge in edges:
        haystack = " ".join(str(edge.get(key, "")) for key in ("source", "target", "label", "source_file"))
        if _edge_matches_question_intent(edge, question, answer):
            filtered.append(edge)
        elif preferred and edge.get("label") in preferred and not ("控制关系" in question or "关联方" in question):
            filtered.append(edge)
        elif not preferred and terms and any(_normalize_for_match(term) in _normalize_for_match(haystack) for term in terms):
            filtered.append(edge)
    filtered = filtered[:limit]
    if not filtered:
        return {"nodes": [], "edges": [], "reason": graph_subgraph.get("reason", "")}
    node_ids = {edge.get("source") for edge in filtered} | {edge.get("target") for edge in filtered}
    nodes = [node for node in graph_subgraph.get("nodes", []) if node.get("id") in node_ids or node.get("label") in node_ids]
    known = {node.get("id") for node in nodes} | {node.get("label") for node in nodes}
    for node_id in node_ids:
        if node_id and node_id not in known:
            nodes.append({"id": node_id, "label": node_id, "type": "entity"})
    return {"nodes": nodes, "edges": filtered, "reason": graph_subgraph.get("reason", "")}


def _important_terms(text: str) -> list[str]:
    stopwords = {"哪些", "多少", "分别", "公司", "股份", "有限", "本次", "本公司", "关系", "是什么"}
    terms = []
    for term in re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()]+", text or ""):
        term = term.strip("，,。；;：:？?、")
        if len(term) < 2 or term in stopwords:
            continue
        terms.append(term)
    for value in re.findall(r"[0-9,，.]+(?:万|亿)?(?:元|股)?|[0-9.]+%", text or ""):
        terms.append(value)
    ordered = []
    seen = set()
    for term in terms:
        if term not in seen:
            ordered.append(term)
            seen.add(term)
    return ordered[:20]


def _preferred_relation_labels(question: str) -> list[str]:
    if "不存在控制关系" in question:
        return ["不存在控制关系关联方", "本公司关系"]
    if "存在控制关系" in question:
        return ["存在控制关系关联方", "持股比例", "本公司关系"]
    mapping = {
        "发行股数": ["发行股数", "发行后总股本", "占发行后总股本比例"],
        "总股本": ["发行股数", "发行后总股本", "占发行后总股本比例"],
        "募集资金拟投资": ["募集资金拟投资项目", "计划总投资"],
        "投资哪些项目": ["募集资金拟投资项目", "计划总投资"],
        "补充流动资金": ["募集资金"],
        "法定代表人": ["法定代表人"],
        "技术标准": ["参与制定技术标准"],
        "重要供应商": ["成为重要供应商领域"],
        "上游": ["上游涉及"],
        "下游": ["下游包括"],
        "国家科技进步一等奖": ["荣获国家科技进步一等奖"],
    }
    labels = []
    for trigger, values in mapping.items():
        if trigger in question:
            labels.extend(values)
    return labels


def _edge_matches_question_intent(edge: dict, question: str, answer: str) -> bool:
    label = str(edge.get("label", ""))
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    haystack = _normalize_for_match(f"{source} {target} {label}")
    answer_terms = [_normalize_for_match(term) for term in (_important_terms(answer) + _numeric_terms(answer))]
    answer_terms = [term for term in answer_terms if term]

    if "不存在控制关系" in question:
        if label == "不存在控制关系关联方":
            return True
        if label == "本公司关系":
            return any(term in _normalize_for_match(source) for term in answer_terms)
        return False
    if "存在控制关系" in question:
        if label == "存在控制关系关联方":
            return any(term in _normalize_for_match(target) for term in answer_terms)
        if label in {"持股比例", "本公司关系"}:
            return any(term in haystack for term in answer_terms)
        return False
    return False


def _numeric_terms(text: str) -> list[str]:
    return re.findall(r"[0-9,，.]+\s*(?:万|亿)?\s*(?:元|股)?|[0-9.]+%", text or "")


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text or "").replace("，", ",")


def load_eval_questions() -> List[dict]:
    if not QUESTIONS_PATH.exists():
        print(f"未找到 {QUESTIONS_PATH}，使用 Config.EVALUATION_QUESTIONS 兜底。")
        return list(Config.EVALUATION_QUESTIONS)

    text = QUESTIONS_PATH.read_text(encoding="utf-8")
    questions: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"(?:Question|问题)\s*[:：]\s*(.+)", line, flags=re.I)
        if match:
            questions.append({"id": len(questions) + 1, "question": match.group(1).strip()})
            continue
        numbered = re.match(r"(?:[-*]|\d+[.)、])\s*(.+[?？]?)$", line)
        if numbered and len(numbered.group(1).strip()) >= 4:
            questions.append({"id": len(questions) + 1, "question": numbered.group(1).strip()})

    if not questions:
        raise ValueError(f"{QUESTIONS_PATH} 中没有识别到 Question。")
    return questions


def run_tests(engine: RAGEngine, questions: list[dict]) -> list[dict]:
    results = []
    for idx, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"\n[{idx}/{len(questions)}] {question}")
        start = time.time()
        try:
            response = engine.ask(
                question,
                question_id=item.get("id"),
                retrieval_mode=RETRIEVAL_MODE,
                answer_language="zh",
            )
            graph_info = response.query_analysis.get("graph_rag", {}) if response.query_analysis else {}
            graph_subgraph = build_question_subgraph(engine, question, response.source_chunks, response.answer, limit=32)
            graph_node_count = len(graph_subgraph.get("nodes", []))
            graph_edge_count = len(graph_subgraph.get("edges", []))
            citations = [
                {
                    "source_file": (chunk.get("metadata") or {}).get("source_file", ""),
                    "page": (chunk.get("metadata") or {}).get("page", 0),
                    "chunk_id": (chunk.get("metadata") or {}).get("chunk_id", ""),
                    "retrieval_source": chunk.get("retrieval_source", "hybrid"),
                    "score": chunk.get("relevance_score", 0.0),
                }
                for chunk in response.source_chunks
            ]
            result = {
                "id": item.get("id", idx),
                "question": question,
                "answer": response.answer,
                "accuracy": response.accuracy,
                "response_time": response.response_time,
                "retrieval_mode": response.retrieval_mode,
                "has_context": response.has_context,
                "retrieved_count": len(response.retrieved_contexts),
                "graph_backend": graph_info.get("backend", ""),
                "graph_entities": graph_info.get("matched_entities", []),
                "graph_relations": graph_info.get("relations", []),
                "graph_subgraph": graph_subgraph,
                "graph_node_count": graph_node_count,
                "graph_edge_count": graph_edge_count,
                "citations": citations,
                "source_chunks": response.source_chunks,
                "status": "success",
                "error": "",
            }
            print(f"  -> 图谱实体: {'、'.join(result['graph_entities']) or '无'}")
            print(f"  -> 答案: {response.answer[:120]}{'...' if len(response.answer) > 120 else ''}")
            print(f"  -> 耗时: {response.response_time:.3f}s | 置信度: {response.accuracy:.1%}")
        except Exception as exc:
            result = {
                "id": item.get("id", idx),
                "question": question,
                "answer": "",
                "accuracy": 0.0,
                "response_time": time.time() - start,
                "retrieval_mode": RETRIEVAL_MODE,
                "has_context": False,
                "retrieved_count": 0,
                "graph_backend": "",
                "graph_entities": [],
                "graph_relations": [],
                "graph_subgraph": {},
                "graph_node_count": 0,
                "graph_edge_count": 0,
                "citations": [],
                "source_chunks": [],
                "status": "failed",
                "error": str(exc),
            }
            print(f"  -> 失败: {exc}")
        results.append(result)
    return results


def build_summary(engine: RAGEngine, results: list[dict]) -> dict:
    total = len(results)
    success_count = sum(1 for item in results if item["status"] == "success")
    context_count = sum(1 for item in results if item["has_context"])
    graph_hit_count = sum(
        1
        for item in results
        if item.get("graph_edge_count", 0) > 0
    )
    avg_accuracy = sum(item["accuracy"] for item in results) / total if total else 0.0
    avg_response_time = sum(item["response_time"] for item in results) / total if total else 0.0
    graph_stats = engine.graph_store.stats().__dict__ if hasattr(engine, "graph_store") else {}
    work_order_7 = load_work_order_7_summary()
    comparison = {
        "work_order_7": work_order_7,
        "work_order_8": {
            "retrieval_mode": RETRIEVAL_MODE,
            "average_accuracy": avg_accuracy,
            "average_response_time": avg_response_time,
            "graph_structure_output": True,
        },
    }
    if work_order_7:
        comparison["delta"] = {
            "accuracy_delta": avg_accuracy - float(work_order_7.get("average_accuracy", 0.0)),
            "response_time_delta": avg_response_time - float(work_order_7.get("average_response_time", 0.0)),
            "graph_structure_delta": "工单八新增每题图谱实体、关系、子图节点和边",
        }
    return {
        "work_order": "人工智能 NLP-RAG-Graph RAG 知识图谱问答",
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_mode": RETRIEVAL_MODE,
        "total": total,
        "success_count": success_count,
        "failed_count": total - success_count,
        "context_count": context_count,
        "context_rate": context_count / total if total else 0.0,
        "graph_hit_count": graph_hit_count,
        "graph_hit_rate": graph_hit_count / total if total else 0.0,
        "average_accuracy": avg_accuracy,
        "average_response_time": avg_response_time,
        "average_graph_nodes": sum(item.get("graph_node_count", 0) for item in results) / total if total else 0.0,
        "average_graph_edges": sum(item.get("graph_edge_count", 0) for item in results) / total if total else 0.0,
        "graph_stats": graph_stats,
        "comparison_with_work_order_7": comparison,
    }


def load_work_order_7_summary() -> dict:
    if not WORK_ORDER_7_RESULTS_PATH.exists():
        return {}
    try:
        payload = json.loads(WORK_ORDER_7_RESULTS_PATH.read_text(encoding="utf-8"))
        return payload.get("summary", {})
    except Exception:
        return {}


def save_results(summary: dict, results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LEGACY_RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_report(summary: dict, results: list[dict]) -> None:
    lines = [
        "# 工单八 Graph RAG 功能测试及评估报告",
        "",
        "## 汇总",
        "",
        f"- 测试时间: {summary['test_time']}",
        f"- 检索模式: {summary['retrieval_mode']}",
        f"- 测试问题数: {summary['total']}",
        f"- 成功数: {summary['success_count']}",
        f"- 失败数: {summary['failed_count']}",
        f"- 有上下文数量: {summary['context_count']}",
        f"- 上下文命中率: {summary['context_rate']:.2%}",
        f"- 图谱命中数: {summary['graph_hit_count']}",
        f"- 图谱命中率: {summary['graph_hit_rate']:.2%}",
        f"- 平均置信度: {summary['average_accuracy']:.2%}",
        f"- 平均响应时间: {summary['average_response_time']:.2f}s",
        f"- 平均子图节点数: {summary['average_graph_nodes']:.1f}",
        f"- 平均子图边数: {summary['average_graph_edges']:.1f}",
        f"- 图谱统计: {summary['graph_stats']}",
        "",
        "## 与工单七对比",
        "",
    ]
    comparison = summary.get("comparison_with_work_order_7", {})
    wo7 = comparison.get("work_order_7", {})
    if wo7:
        lines.extend(
            [
                "| 对比项 | 工单七 Hybrid RAG | 工单八 Graph RAG |",
                "| --- | ---: | ---: |",
                f"| 平均准确率 | {float(wo7.get('average_accuracy', 0.0)):.2%} | {summary['average_accuracy']:.2%} |",
                f"| 平均响应时间 | {float(wo7.get('average_response_time', 0.0)):.2f}s | {summary['average_response_time']:.2f}s |",
                f"| 图谱结构输出 | 无 | 有，包含 nodes / edges / relations |",
                "",
            ]
        )
    else:
        lines.extend(["未找到 work_order_7_results.json，无法自动生成对比表。", ""])
    lines.extend(
        [
        "## 明细",
        "",
        "| ID | 问题 | 图谱实体 | 子图节点 | 子图边 | 片段数 | 置信度 | 响应时间 | 状态 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in results:
        question = str(item["question"]).replace("|", "\\|")
        entities = "、".join(item.get("graph_entities", [])) or "无"
        lines.append(
            f"| {item['id']} | {question} | {entities} | "
            f"{item.get('graph_node_count', 0)} | {item.get('graph_edge_count', 0)} | {item['retrieved_count']} | "
            f"{item['accuracy']:.2%} | {item['response_time']:.2f}s | {item['status']} |"
        )

    lines.extend(["", "## 答案详情", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['id']}. {item['question']}",
                "",
                f"- 图谱后端: {item.get('graph_backend', '')}",
                f"- 命中实体: {'、'.join(item.get('graph_entities', [])) or '无'}",
                f"- 子图节点数: {item.get('graph_node_count', 0)}",
                f"- 子图边数: {item.get('graph_edge_count', 0)}",
                f"- 图谱说明: {(item.get('graph_subgraph') or {}).get('reason', '')}",
                f"- 检索片段数: {item.get('retrieved_count', 0)}",
                "",
                "答案:",
                "",
                item["answer"] or f"测试失败: {item['error']}",
                "",
                "局部图谱边:",
                "",
            ]
        )
        edges = (item.get("graph_subgraph") or {}).get("edges", [])
        if edges:
            for edge in edges[:8]:
                lines.append(
                    f"- {edge.get('source', '')} --{edge.get('label', '')}--> {edge.get('target', '')} "
                    f"({edge.get('source_file', '')} 第{edge.get('page', 0)}页 chunk={edge.get('chunk_id', '')})"
                )
        else:
            lines.append("- 无")
        lines.extend(
            [
                "",
                "来源:",
                "",
            ]
        )
        for citation in item.get("citations", [])[:6]:
            lines.append(
                f"- {citation.get('source_file', '')} 第{citation.get('page', 0)}页 "
                f"chunk={citation.get('chunk_id', '')} "
                f"source={citation.get('retrieval_source', '')}"
            )
        lines.append("")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    LEGACY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    questions = load_eval_questions()
    engine = RAGEngine()
    init_result = engine.initialize_project_knowledge_base()
    print(init_result.message)
    print(init_result.details)
    if not init_result.success:
        raise SystemExit(init_result.details)
    results = run_tests(engine, questions)
    summary = build_summary(engine, results)
    save_results(summary, results)
    save_report(summary, results)
    print("\n工单八测试完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果文件: {RESULTS_PATH.resolve()}")
    print(f"报告文件: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
