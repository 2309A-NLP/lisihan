# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from src.constants import NEGATIVE_QUERY_FALLBACK
from src.validation.answer_validator import validate_answer_quality
from src.validation.negative_handler import handle_negative_query

from .engine import RAGEngine

__all__ = [
    "RAGEngine",
    "NEGATIVE_QUERY_FALLBACK",
    "validate_answer_quality",
    "handle_negative_query",
]
