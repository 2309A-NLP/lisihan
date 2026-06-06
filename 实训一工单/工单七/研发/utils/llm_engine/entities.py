# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re


ENTITY_EXTRACT_MAX_CHARS = 500


def _normalize_entity_source(chunk: str) -> str:
    text = re.sub(r"\s+", " ", chunk or "")
    text = re.sub(r"\|+", " ", text)
    return text.strip()


def extract_complete_entity(chunk: str, entity_type: str) -> str:
    """
    完整提取实体信息，不被截断。
    对于法定代表人：提取"法定代表人：XXX"完整模式。
    对于注册资本：提取数字+万元/亿元模式。
    最大提取窗口为 500 字符。
    """
    text = _normalize_entity_source(chunk)[:ENTITY_EXTRACT_MAX_CHARS]
    entity_type = (entity_type or "").strip().lower()

    if entity_type in {"legal_representative", "法定代表人"}:
        match = re.search(
            r"法定代表人\s*[:：]?\s*([\u4e00-\u9fa5A-Za-z·]{2,20})",
            text,
        )
        if match:
            return f"法定代表人：{match.group(1)}"
        return ""

    if entity_type in {"registered_capital", "注册资本"}:
        match = re.search(
            r"注册资本\s*[:：]?\s*([0-9０-９,，]+(?:\.\d+)?\s*(?:万元|亿元|万人民币|人民币万元|人民币亿元))",
            text,
        )
        if match:
            value = match.group(1).replace("，", ",").replace(" ", "").replace(".00", "")
            return f"注册资本：{value}"
        match = re.search(r"([0-9０-９,，]+(?:\.\d+)?\s*(?:万元|亿元))", text)
        if match:
            value = match.group(1).replace("，", ",").replace(" ", "").replace(".00", "")
            return f"注册资本：{value}"
        return ""

    return text[:ENTITY_EXTRACT_MAX_CHARS]
