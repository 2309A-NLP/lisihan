# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from typing import Dict, Tuple

from pdf_parser.main import PDFParser
from src.cache.metadata_cache import MetadataCache
from src.config import Config
from src.memory_manager import LongTermMemoryManager
from src.models import InitResult, RAGResponse
from src.multimodal.image_parser import MultimodalImageParser
from src.retriever import HybridRetriever
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
        self.pdf_parser = PDFParser(
            chunk_size=Config.PDF_CHUNK_SIZE,
            chunk_overlap=Config.PDF_CHUNK_OVERLAP,
            chunk_strategy=Config.PDF_CHUNK_STRATEGY,
            parent_chunk_size=Config.PDF_PARENT_CHUNK_SIZE,
        )
        self.vector_store = HybridRetriever()
        self.long_term_memory = LongTermMemoryManager()
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
