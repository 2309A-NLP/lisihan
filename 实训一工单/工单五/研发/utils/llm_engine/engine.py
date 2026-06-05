# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from src.config import Config
from utils.logger import get_logger
from utils.question_classifier import QuestionClassifier

from .generation import GenerationMixin
from .local_answer import LocalAnswerMixin
from .numeric import NumericExtractionMixin
from .text_tools import TextToolsMixin

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - 可选依赖
    OpenAI = None


logger = get_logger(__name__)


class LLMEngine(TextToolsMixin, NumericExtractionMixin, LocalAnswerMixin, GenerationMixin):
    FALLBACK_MODELS = []

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.LLM_MODEL
        self.question_classifier = QuestionClassifier()
        self.client = self._init_client()

    def _init_client(self):
        if OpenAI is None or not Config.LLM_API_KEY:
            logger.info("openai disabled | reason=no_client_or_key")
            return None
        try:
            return OpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_API_BASE_URL,
                timeout=Config.LLM_TIMEOUT,
                max_retries=0,
            )
        except Exception:
            logger.exception("openai client init failed")
            return None
