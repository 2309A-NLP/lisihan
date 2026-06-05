# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InitResult:
    success: bool
    status: str
    message: str
    details: str = ""
    document_count: int = 0


@dataclass
class RAGResponse:
    question: str
    answer: str
    question_type: str
    memory_hit: bool
    retrieval_mode: str
    retrieved_contexts: List[str]
    scores: List[float]
    response_time: float
    accuracy: float
    source_chunks: List[Dict]
    has_context: bool
    query_analysis: Dict
