# -*- coding: utf-8 -*-
# 人工智能 NLP-RAG-基于 PDF 文档的问答系统
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""Data models for the PDF RAG QA system."""

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
