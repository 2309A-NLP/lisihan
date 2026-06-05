# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from pdf_parser.main import render_pdf_page_to_image
from src.config import Config
from src.constants import NEGATIVE_QUERY_FALLBACK
from src.models import InitResult, RAGResponse
from src.processing.query_rewriter import _is_liyuan_document, rewrite_query, select_pdf
from src.processing.table_extractor import _extract_liyuan_table_answer
from src.utils.text_utils import _normalize_question_text
from src.validation.answer_validator import validate_answer_quality
from src.validation.negative_handler import handle_negative_query
from utils.llm_engine import extract_complete_entity

class InitializationMixin:
    def initialize(self, pdf_path: str = None) -> bool:
        if self.vector_store.load_vectorstore():
            self.is_initialized = True
            return True
        if pdf_path:
            documents = self.pdf_parser.parse_pdf(pdf_path)
            if documents:
                self.vector_store.create_vectorstore(documents)
                self.is_initialized = True
                return True
        return False

    def initialize_from_dir(self, pdf_dir: str = None) -> bool:
        pdf_dir = pdf_dir or Config.PDF_DIR
        documents = self.pdf_parser.parse_multiple_pdfs(pdf_dir)
        if documents:
            self.vector_store.create_vectorstore(documents)
            self._refresh_document_metadata()
            self.is_initialized = True
            return True
        return False

    def initialize_from_project(self) -> bool:
        return self.initialize_project_knowledge_base().success

    def initialize_project_knowledge_base(self) -> InitResult:
        try:
            self.logger.info("init knowledge base start | pdf_dir=%s", Config.PDF_DIR)
            documents = self.pdf_parser.parse_multiple_pdfs(Config.PDF_DIR)
            if not documents:
                self.is_initialized = False
                self.last_init_result = InitResult(
                    False,
                    "no_pdf",
                    "项目中没有可解析的PDF文档。",
                    details=f"请把 PDF 放到 {Config.PDF_DIR} 目录。",
                )
                self.logger.warning("no pdf documents found | pdf_dir=%s", Config.PDF_DIR)
                return self.last_init_result

            self.vector_store.create_vectorstore(documents)
            self._refresh_document_metadata()
            self.is_initialized = True
            self.last_init_result = InitResult(
                True,
                "indexed",
                "已解析项目 PDF，并构建混合检索索引。",
                details=f"index={Config.COLLECTION_NAME}",
                document_count=len(documents),
            )
            self.logger.info("knowledge base indexed | documents=%s", len(documents))
            return self.last_init_result
        except Exception as e:
            self.is_initialized = False
            self.logger.exception("knowledge base init failed")
            self.last_init_result = InitResult(
                False,
                "init_error",
                "项目知识库初始化失败。",
                details=str(e),
            )
            return self.last_init_result
