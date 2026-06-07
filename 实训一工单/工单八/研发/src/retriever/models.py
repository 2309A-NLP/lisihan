# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.document import Document


@dataclass
class _StoredDocument:
    doc: Document
    tokens: List[str]


@dataclass
class _RetrievalHit:
    doc: Document
    score: float
    auto_fallback: bool = False
