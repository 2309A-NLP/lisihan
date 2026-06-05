# -*- coding: utf-8 -*-
# 人工智能 NLP-RAG-基于 PDF 文档的问答系统
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""Text normalization and extraction helpers."""

from __future__ import annotations

import re
from typing import List


def _normalize_question_text(text: str) -> str:
    """Normalize user questions while preserving meaningful English spacing."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    return normalized


def _chunk_to_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        parts = []
        for key in ("content", "page_content", "text", "question", "answer"):
            value = chunk.get(key)
            if value:
                parts.append(str(value))
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            parts.extend(str(value) for value in metadata.values() if value)
        elif metadata:
            parts.append(str(metadata))
        return " ".join(parts)
    return str(chunk or "")


def _extract_company_entities(text: str) -> List[str]:
    entities = re.findall(r"[\u4e00-\u9fff]{2,}(?:股份有限公司|有限责任公司|有限公司|公司)", text or "")
    aliases = []
    for keyword in ["武汉力源信息技术股份有限公司", "力源信息", "武汉兴图新科电子股份有限公司", "兴图新科"]:
        if keyword in (text or ""):
            aliases.append(keyword)
    seen = set()
    return [entity for entity in [*entities, *aliases] if not (entity in seen or seen.add(entity))]


def _clean_candidate_name(text: str) -> str:
    name = re.sub(r"\s+", "", text or "")
    name = name.strip("|:：,，;；。 ")
    return name


def _is_valid_related_party_name(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 24:
        return False
    skip_terms = {
        "关联方名称",
        "关联人名称",
        "企业名称",
        "与发行人的关系",
        "与本公司关系",
        "持股比例",
        "项目名称",
        "公司地址",
        "证券代码",
        "证券简称",
    }
    if name in skip_terms or set(name) <= {"-", "—"}:
        return False
    if re.search(r"\d", name) and not re.search(r"[\u4e00-\u9fff]", name):
        return False
    if any(term in name for term in ["持有", "控制", "实际控制人", "董事", "监事", "高管", "关联关系", "注册资本", "法定代表人"]):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", name))
