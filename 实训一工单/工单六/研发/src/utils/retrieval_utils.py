# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from typing import List


def _safe_retriever_search(
    retriever,
    query: str,
    *,
    top_k: int = 8,
    source_file: str | None = None,
    mode: str = "bm25",
) -> List:
    try:
        return retriever.search(query, top_k=top_k, mode=mode, source_file=source_file)
    except TypeError:
        try:
            return retriever.search(query, top_k=top_k, mode=mode)
        except TypeError:
            return retriever.search(query, top_k)
    except Exception:
        return []


def _doc_from_retrieval_item(item):
    if isinstance(item, tuple) and item:
        return item[0]
    return item


def _doc_source_file(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return metadata.get("source_file", "")


def _doc_page(doc) -> int:
    metadata = getattr(doc, "metadata", {}) or {}
    try:
        return int(metadata.get("page", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _doc_chunk_id(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("chunk_id") or getattr(doc, "metadata", {}).get("id", ""))


def _iter_indexed_docs(retriever, source_file: str | None = None) -> List:
    docs = []
    for item in getattr(retriever, "documents", []) or []:
        doc = getattr(item, "doc", item)
        if source_file and _doc_source_file(doc) != source_file:
            continue
        docs.append(doc)
    return docs


def _neighbor_docs_after_marker(retriever, source_file: str | None, marker_predicate, window: int = 2) -> List:
    docs = _iter_indexed_docs(retriever, source_file=source_file)
    selected = []
    seen = set()
    for idx, doc in enumerate(docs):
        content = getattr(doc, "page_content", "") or ""
        if not marker_predicate(content):
            continue
        for near_doc in docs[idx : idx + window + 1]:
            key = (_doc_source_file(near_doc), _doc_page(near_doc), _doc_chunk_id(near_doc), getattr(near_doc, "page_content", "")[:80])
            if key in seen:
                continue
            selected.append(near_doc)
            seen.add(key)
    return selected
