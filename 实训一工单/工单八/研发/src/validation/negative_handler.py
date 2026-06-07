# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import List, Sequence

from src.constants import NEGATIVE_QUERY_FALLBACK
from src.processing.query_rewriter import _infer_source_file_from_question
from src.utils.retrieval_utils import (
    _doc_from_retrieval_item,
    _neighbor_docs_after_marker,
    _safe_retriever_search,
)
from src.utils.text_utils import _chunk_to_text, _clean_candidate_name, _is_valid_related_party_name, _normalize_question_text


def _is_negative_question(question: str) -> bool:
    q = _normalize_question_text(question)
    return any(marker in q for marker in ["不存在", "无", "未", "未披露", "没有", "不具有", "非", "除外"]) or bool(
        re.search(r"除了.+之外", q)
    )


def _looks_like_negative_related_query(question: str) -> bool:
    q = _normalize_question_text(question)
    return _is_negative_question(q) and any(term in q for term in ["关联方", "关联关系", "控制关系"])


def _extract_related_party_names_from_text(text: str) -> List[str]:
    names: List[str] = []
    for raw_line in re.split(r"[\r\n]+", text or ""):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|") and "|" in line:
            cells = [_clean_candidate_name(cell) for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if not cells or any(set(cell) <= {"-", "—"} for cell in cells):
                continue
            candidate = cells[0]
            if _is_valid_related_party_name(candidate):
                names.append(candidate)
            continue
        for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9（）()]{2,24}", line):
            candidate = _clean_candidate_name(match)
            if _is_valid_related_party_name(candidate):
                names.append(candidate)
    seen = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def _extract_related_party_names(chunks: Sequence) -> List[str]:
    names: List[str] = []
    for chunk in chunks or []:
        doc = _doc_from_retrieval_item(chunk)
        content = getattr(doc, "page_content", None)
        if content is None:
            content = _chunk_to_text(chunk)
        names.extend(_extract_related_party_names_from_text(content))
    seen = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def handle_negative_query(question: str, retriever) -> list:
    """
    处理否定类查询。
    1. 识别否定词：不存在、没有、未披露、除外、除了...之外
    2. 先检索正面答案（存在控制关系的关联方）
    3. 再从全量关联方中排除正面答案，得到否定答案
    4. 如果找不到明确答案，返回"招股说明书中未披露相关信息"
    """
    question = _normalize_question_text(question)
    if not _looks_like_negative_related_query(question):
        return []
    if "未披露" in (question or ""):
        return [NEGATIVE_QUERY_FALLBACK]

    source_file = _infer_source_file_from_question(question)
    positive_docs = _neighbor_docs_after_marker(
        retriever,
        source_file,
        lambda content: "存在控制关系的关联方" in content and "不存在控制关系的关联方" not in content,
        window=1,
    )
    if not positive_docs:
        positive_hits = _safe_retriever_search(
            retriever,
            "存在控制关系的关联方 关联方名称 持股比例 与本公司关系",
            top_k=4,
            source_file=source_file,
            mode="bm25",
        )
        positive_docs = [
            _doc_from_retrieval_item(item)
            for item in positive_hits
            if "不存在控制关系的关联方" not in (getattr(_doc_from_retrieval_item(item), "page_content", "") or "")
            and "企业名称" not in (getattr(_doc_from_retrieval_item(item), "page_content", "") or "")
        ]
    positive_names = set(_extract_related_party_names(positive_docs))

    negative_hits = []
    if any(term in (question or "") for term in ["不存在控制关系", "无控制关系", "没有控制关系", "不具有控制关系"]):
        negative_hits = _neighbor_docs_after_marker(
            retriever,
            source_file,
            lambda content: "不存在控制关系的关联方" in content,
            window=2,
        )

    all_hits = _safe_retriever_search(
        retriever,
        "关联方及关联关系 关联方名称 企业名称 与本公司关系",
        top_k=16,
        source_file=source_file,
        mode="bm25",
    )
    all_docs = [_doc_from_retrieval_item(item) for item in all_hits]
    all_docs.extend(
        _neighbor_docs_after_marker(
            retriever,
            source_file,
            lambda content: any(marker in content for marker in ["关联方及关联关系", "关联方名称", "企业名称"]),
            window=2,
        )
    )

    negative_section_names = _extract_related_party_names(negative_hits)
    candidate_names = negative_section_names or _extract_related_party_names(all_docs)
    if not candidate_names:
        return [NEGATIVE_QUERY_FALLBACK]

    excluded_keywords = set()
    outside_match = re.search(r"除了(.+?)之外", question or "")
    if outside_match:
        excluded_keywords.update(_extract_related_party_names_from_text(outside_match.group(1)))
    excluded_keywords.update(positive_names)

    negative_names = [name for name in candidate_names if name not in excluded_keywords]
    if not negative_names:
        return [NEGATIVE_QUERY_FALLBACK]
    return negative_names
