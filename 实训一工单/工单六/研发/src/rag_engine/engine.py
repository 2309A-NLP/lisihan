# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件属于 PDF 招股说明书智能问答系统，用于接入工单五多轮对话与指代消解，
并保留工单一到工单四的文本检索、图片内容解析和 Redis 缓存能力。
"""

from __future__ import annotations

from typing import Dict, Tuple

from pdf_parser.main import PDFParser
from src.cache.metadata_cache import MetadataCache
from src.config import Config
from src.coreference_resolver import CoreferenceResolver
from src.memory_manager import LongTermMemoryManager
from src.models import InitResult, RAGResponse
from src.multimodal.image_parser import MultimodalImageParser
from src.retriever import HybridRetriever
from src.session_manager import SessionManager
from utils.llm_engine import LLMEngine
from utils.logger import get_logger
from utils.question_classifier import QuestionClassifier

from .answering import AnsweringMixin
from .cache import AnswerCacheMixin
from .context import ContextMixin
from .initializer import InitializationMixin
from .metadata import MetadataMixin
from .multimodal import MultimodalMixin


class RAGEngine(
    MetadataMixin,
    AnswerCacheMixin,
    InitializationMixin,
    ContextMixin,
    MultimodalMixin,
    AnsweringMixin,
):
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pdf_parser = PDFParser()
        self.vector_store = HybridRetriever()
        self.long_term_memory = LongTermMemoryManager()
        self.session_manager = SessionManager()
        self.coreference_resolver = CoreferenceResolver()
        self.llm_engine = LLMEngine()
        self.image_parser = MultimodalImageParser(
            Config.MULTIMODAL_API_KEY,
            Config.MULTIMODAL_MODEL,
            Config.MULTIMODAL_API_BASE_URL,
        )
        self.question_classifier = QuestionClassifier()
        self.is_initialized = False
        self.last_init_result = InitResult(False, "not_started", "系统尚未初始化")
        self.metadata_cache = MetadataCache(self.pdf_parser, self.logger)
        self._answer_cache: Dict[str, Tuple[RAGResponse, float]] = {}
