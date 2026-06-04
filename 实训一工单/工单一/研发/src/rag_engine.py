# -*- coding: utf-8 -*-
"""RAG engine for the PDF QA system."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.config import Config
from src.pdf_parser import PDFParser
from src.vector_store import VectorStore
from utils.llm_engine import LLMEngine
from utils.logger import get_logger
from utils.question_classifier import QuestionClassifier


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
    retrieval_time: float
    query_time: float
    generation_time: float
    total_time: float
    source_chunks: List[Dict]
    has_context: bool
    query_analysis: Dict


class RAGEngine:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pdf_parser = PDFParser()
        self.vector_store = VectorStore()
        self.llm_engine = LLMEngine()
        self.question_classifier = QuestionClassifier()
        self.is_initialized = False
        self.last_init_result = InitResult(False, "not_started", "系统尚未初始化")

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
            self.is_initialized = True
            self.last_init_result = InitResult(
                True,
                "indexed",
                "已解析项目 PDF，并构建 BM25 检索索引。",
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

    def _build_context(self, question: str, top_k: int = 4, question_id: int = None):
        analysis = self.llm_engine.query_intent(question, question_id=question_id)
        retrieved_results = self.vector_store.search(question, top_k, mode="bm25")

        context_parts = []
        scores = []
        source_chunks = []
        for doc, score in retrieved_results:
            context_parts.append(doc.page_content)
            scores.append(score)
            source_chunks.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        return analysis, context, context_parts, scores, source_chunks

    def ask(self, question: str, question_id: int = None, retrieval_mode: str = "bm25", session_id: str = "default") -> RAGResponse:
        start_time = time.perf_counter()
        question_info = self.question_classifier.classify(question, question_id=question_id)
        question_type = question_info["question_type"]
        prompt_type = "entity" if question_type == "entity" else "general"
        self.logger.info("ask start | question=%s | question_type=%s", question, question_type)

        retrieval_mode = "bm25"
        query_analysis, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
        )
        retrieval_time = time.perf_counter() - start_time
        has_context = len(context_parts) > 0
        query_time = 0.0
        generation_time = 0.0
        if has_context:
            query_start = time.perf_counter()
            answer = self.llm_engine.generate_answer(question, context, prompt_type=prompt_type)
            query_time = time.perf_counter() - query_start
            generation_start = time.perf_counter()
            if question_type in {"numeric", "percentage"}:
                extracted = self.llm_engine._extract_answer_from_context(question, context)
                if extracted:
                    answer = extracted
            generation_time = time.perf_counter() - generation_start
        else:
            generation_start = time.perf_counter()
            answer = "根据当前知识库，暂时没有找到足够相关的信息。"
            generation_time = time.perf_counter() - generation_start

        response_time = time.perf_counter() - start_time
        self.logger.info(
            "ask done | question=%s | question_type=%s | intent=%s | ambiguous=%s | has_context=%s | hits=%s | elapsed=%.3f",
            question,
            question_type,
            query_analysis.get("intent"),
            query_analysis.get("is_ambiguous"),
            has_context,
            len(context_parts),
            response_time,
        )
        return RAGResponse(
            question=question,
            answer=answer,
            question_type=question_type,
            memory_hit=False,
            retrieval_mode=retrieval_mode,
            retrieved_contexts=context_parts,
            scores=scores,
            response_time=response_time,
            retrieval_time=retrieval_time,
            query_time=query_time,
            generation_time=generation_time,
            total_time=response_time,
            source_chunks=source_chunks,
            has_context=has_context,
            query_analysis=query_analysis,
        )

    def answer(self, question: str, top_k: int = None, question_id: int = None) -> RAGResponse:
        return self.ask(question, question_id=question_id)

    def stream_answer(self, question: str, top_k: int = None, question_id: int = None, retrieval_mode: str = "bm25"):
        question_info = self.question_classifier.classify(question, question_id=question_id)
        prompt_type = "entity" if question_info["question_type"] == "entity" else "general"
        _, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
        )
        has_context = len(context_parts) > 0
        if not has_context:
            yield "根据当前知识库，暂时没有找到足够相关的信息。"
            return

        for chunk in self.llm_engine.stream_answer(question, context, prompt_type=prompt_type):
            yield chunk

    def answer_without_rag(self, question: str) -> Tuple[str, float]:
        start_time = time.time()
        answer = self.llm_engine.generate_answer_without_context(question)
        response_time = time.time() - start_time
        self.logger.info("answer_without_rag done | question=%s | elapsed=%.3f", question, response_time)
        return answer, response_time

    def record_feedback(self, question: str, answer: str, helpful: bool, question_type: str = "entity") -> None:
        self.logger.info("feedback recorded | helpful=%s | question=%s", helpful, question)

    def batch_answer(self, questions: List[Dict]) -> List[Dict]:
        results = []
        for q in questions:
            response = self.ask(q["question"], question_id=q.get("id"))
            answer_no_rag, time_no_rag = self.answer_without_rag(q["question"])
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "rag_answer": response.answer,
                    "question_type": response.question_type,
                    "retrieval_mode": response.retrieval_mode,
                    "rag_response_time": response.response_time,
                    "retrieval_time": response.retrieval_time,
                    "query_time": response.query_time,
                    "generation_time": response.generation_time,
                    "total_time": response.total_time,
                    "rag_has_context": response.has_context,
                    "rag_retrieved_count": len(response.retrieved_contexts),
                    "llm_only_answer": answer_no_rag,
                    "llm_only_response_time": time_no_rag,
                    "retrieved_contexts": response.retrieved_contexts,
                    "has_context": response.has_context,
                    "response_time": response.response_time,
                    "query_analysis": response.query_analysis,
                }
            )
        return results

