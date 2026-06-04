# -*- coding: utf-8 -*-
"""RAG engine for the PDF QA system."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from pdf_parser.main import PDFParser
from src.config import Config
from src.memory_manager import LongTermMemoryManager
from src.retriever import HybridRetriever
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
    source_chunks: List[Dict]
    has_context: bool
    query_analysis: Dict


class RAGEngine:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pdf_parser = PDFParser()
        self.vector_store = HybridRetriever()
        self.long_term_memory = LongTermMemoryManager()
        self.llm_engine = LLMEngine()
        self.question_classifier = QuestionClassifier()
        self.is_initialized = False
        self.last_init_result = InitResult(False, "not_started", "系统尚未初始化")
        self.document_metadata: Dict[str, str] = {}

    def _should_skip_long_term_memory(self, question: str, question_type: str) -> bool:
        """精确型招股书问题优先查 PDF，避免首次加载向量模型拖慢响应。"""
        if question_type in {"numeric", "percentage"}:
            return True
        exact_keywords = [
            "技术标准", "上游", "下游", "法定代表人", "注册资本", "募集资金", "重要供应商", "哪个领域",
            "国家科技进步一等奖",
            "technical standard", "upstream", "downstream", "legal representative", "registered capital",
            "working capital", "important supplier", "which field", "national science and technology progress award",
        ]
        q = (question or "").lower()
        return any(keyword in q for keyword in exact_keywords)

    def rewrite_query(self, question: str) -> str:
        """把业务口径问题改写成招股书中的标准字段，提升召回准确率。"""
        q = question or ""
        lower_q = q.lower()
        english_query_map = [
            (["military", "defense"], ["income", "revenue", "sales"], "直接和间接向国防客户的销售额合计分别是多少"),
            (["revenue", "income"], ["proportion", "percentage", "share", "main business"], "收入占主营业务收入的比重分别是多少"),
            (["technical standard"], [], "参与制定了哪个技术标准"),
            (["upstream"], [], "上游涉及哪些企业"),
            (["downstream"], [], "下游主要包括哪些行业"),
            (["important supplier", "which field"], [], "在哪个领域已经成为重要供应商"),
            (["national science and technology progress award"], [], "哪个工程荣获了国家科技进步一等奖"),
            (["registered capital"], [], "注册资本是多少"),
            (["legal representative"], [], "法定代表人是谁"),
            (["working capital", "supplement working capital"], [], "计划使用多少募集资金补充流动资金"),
        ]
        for required_terms, optional_terms, rewritten in english_query_map:
            if all(term in lower_q for term in required_terms) and (
                not optional_terms or any(term in lower_q for term in optional_terms)
            ):
                return rewritten

        is_military_income = any(term in q for term in ["军用领域", "国防客户", "军方客户", "军品业务"])
        is_income_amount = "收入" in q or "销售额" in q
        is_percentage = any(term in q for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"])
        if is_military_income and is_income_amount and not is_percentage:
            return "直接和间接向国防客户的销售额合计分别是多少"
        return q

    def _metadata_answer(self, question: str) -> str:
        q = question or ""
        lower_q = q.lower()
        if not self.document_metadata:
            self._refresh_document_metadata()
        metadata = self.document_metadata
        if "法定代表人" in q or "legal representative" in lower_q:
            return metadata.get("legal_representative", "")
        if "注册资本" in q or "registered capital" in lower_q:
            return metadata.get("registered_capital", "")
        if "成立日期" in q or "成立时间" in q or "establishment date" in lower_q or "founded" in lower_q:
            return metadata.get("establishment_date", "")
        if "注册地址" in q or "注册地" in q or "registered address" in lower_q:
            return metadata.get("registered_address", "")
        if "公司名称" in q or "company name" in lower_q:
            return metadata.get("company_name", "")
        return ""

    def _load_document_metadata_cache(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        root = Path(self.pdf_parser.output_dir)
        if not root.exists():
            return metadata
        for metadata_file in sorted(root.glob("*_metadata.json")):
            try:
                metadata.update(json.loads(metadata_file.read_text(encoding="utf-8")))
            except Exception as exc:
                self.logger.warning("document metadata load failed | file=%s | error=%s", metadata_file, exc)
        return metadata

    def _refresh_document_metadata(self) -> None:
        self.document_metadata.update(getattr(self.pdf_parser, "document_metadata", {}) or {})
        if not self.document_metadata:
            self.document_metadata.update(self._load_document_metadata_cache())

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

    def initialize_from_dir(self, pdf_dir: str = None, *, force_reparse: bool = False) -> bool:
        pdf_dir = pdf_dir or Config.PDF_DIR
        if force_reparse:
            documents = self.pdf_parser.reparse_multiple_pdfs(pdf_dir)
        else:
            documents = self.pdf_parser.load_or_parse_multiple_pdfs(pdf_dir)
        if documents:
            self.vector_store.create_vectorstore(documents)
            self._refresh_document_metadata()
            self.is_initialized = True
            return True
        return False

    def initialize_from_project(self) -> bool:
        return self.initialize_project_knowledge_base().success

    def initialize_project_knowledge_base(self, *, force_reparse: bool = False) -> InitResult:
        try:
            self.logger.info(
                "init knowledge base start | pdf_dir=%s | force_reparse=%s",
                Config.PDF_DIR,
                force_reparse,
            )
            if force_reparse:
                documents = self.pdf_parser.reparse_multiple_pdfs(Config.PDF_DIR)
            else:
                documents = self.pdf_parser.load_parsed_multiple_pdfs(Config.PDF_DIR)

            if not documents:
                self.is_initialized = False
                self.last_init_result = InitResult(
                    False,
                    "no_cache",
                    "未找到可直接加载的 MinerU 解析结果。",
                    details=f"请点击“重新解析并构建混合检索索引”，或确认 {Config.PDF_PARSE_OUTPUT_DIR} 下已有 *_chunks.json。",
                )
                self.logger.warning("no parsed pdf cache found | pdf_dir=%s", Config.PDF_DIR)
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

    def _build_context(self, question: str, top_k: int = 4, question_id: int = None, retrieval_mode: str = "hybrid"):
        analysis = self.llm_engine.query_intent(question, question_id=question_id)
        search_query = self.rewrite_query(question)
        if search_query != question:
            analysis["rewritten_query"] = search_query
        retrieved_results = self.vector_store.search(search_query, top_k, mode=retrieval_mode)

        context_parts = []
        scores = []
        source_chunks = []
        seen_contexts = set()
        for doc, score in retrieved_results:
            metadata = doc.metadata or {}
            parent_content = metadata.get("parent_content", "")
            if parent_content and parent_content != doc.page_content:
                context_text = (
                    f"[父块标题] {metadata.get('parent_title', '')}\n\n"
                    f"[命中子块]\n{doc.page_content}\n\n"
                    f"[父块上下文]\n{parent_content}"
                ).strip()
            else:
                context_text = doc.page_content

            context_key = (metadata.get("source_file"), metadata.get("parent_id"), context_text[:160])
            if context_key in seen_contexts:
                continue
            seen_contexts.add(context_key)

            context_parts.append(context_text)
            scores.append(score)
            source_chunks.append(
                {
                    "content": context_text,
                    "metadata": metadata,
                    "relevance_score": score,
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        analysis["chunk_fusion"] = "child_retrieval_with_parent_context"
        return analysis, context, context_parts, scores, source_chunks

    def ask(
        self,
        question: str,
        question_id: int = None,
        retrieval_mode: str = "hybrid",
        session_id: str = "default",
        answer_language: str = "zh",
    ) -> RAGResponse:
        start_time = time.time()
        question_info = self.question_classifier.classify(question, question_id=question_id)
        question_type = question_info["question_type"]
        prompt_type = question_type if question_type in {"entity", "numeric", "percentage"} else "general"
        self.logger.info("ask start | question=%s | question_type=%s", question, question_type)

        metadata_answer = self._metadata_answer(question)
        if metadata_answer:
            metadata_answer = self.llm_engine.ensure_answer_language(
                question,
                metadata_answer,
                answer_language=answer_language,
            )
            response_time = time.time() - start_time
            return RAGResponse(
                question=question,
                answer=metadata_answer,
                question_type=question_type,
                memory_hit=False,
                retrieval_mode="metadata",
                retrieved_contexts=["document_metadata"],
                scores=[1.0],
                response_time=response_time,
                source_chunks=[{"content": "document_metadata", "metadata": self.document_metadata, "relevance_score": 1.0}],
                has_context=True,
                query_analysis={"intent": "metadata_lookup", "is_ambiguous": False},
            )

        memory_match = None
        if not self._should_skip_long_term_memory(question, question_type):
            memory_match = self.long_term_memory.search(question)
        if memory_match:
            memory_answer = self.llm_engine.ensure_answer_language(
                question,
                memory_match["answer"],
                answer_language=answer_language,
            )
            response_time = time.time() - start_time
            self.logger.info(
                "long-term memory hit | question=%s | score=%.4f | elapsed=%.3f",
                question,
                memory_match["score"],
                response_time,
            )
            return RAGResponse(
                question=question,
                answer=memory_answer,
                question_type=question_type,
                memory_hit=True,
                retrieval_mode="long_term_memory",
                retrieved_contexts=[memory_match["question"]],
                scores=[memory_match["score"]],
                response_time=response_time,
                source_chunks=[{"content": memory_match["question"], "metadata": memory_match, "relevance_score": memory_match["score"]}],
                has_context=True,
                query_analysis={"intent": "long_term_memory", "is_ambiguous": False},
            )

        query_analysis, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
            retrieval_mode=retrieval_mode,
        )
        has_context = len(context_parts) > 0
        if has_context:
            extracted = self.llm_engine._extract_answer_from_context(question, context)
            if extracted and question_type in {"numeric", "percentage", "entity"}:
                answer = self.llm_engine.ensure_answer_language(
                    question,
                    extracted,
                    answer_language=answer_language,
                    context=context,
                )
            else:
                answer = self.llm_engine.ensure_answer_language(
                    question,
                    self.llm_engine.generate_answer(
                        question,
                        context,
                        prompt_type=prompt_type,
                        answer_language=answer_language,
                    ),
                    answer_language=answer_language,
                    context=context,
                )
        else:
            answer = self.llm_engine._no_answer_text(answer_language)

        response_time = time.time() - start_time
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
            source_chunks=source_chunks,
            has_context=has_context,
            query_analysis=query_analysis,
        )

    def answer(self, question: str, top_k: int = None, question_id: int = None, answer_language: str = "zh") -> RAGResponse:
        return self.ask(question, question_id=question_id, answer_language=answer_language)

    def stream_answer(
        self,
        question: str,
        top_k: int = None,
        question_id: int = None,
        retrieval_mode: str = "hybrid",
        answer_language: str = "zh",
    ):
        question_info = self.question_classifier.classify(question, question_id=question_id)
        question_type = question_info["question_type"]
        prompt_type = question_type if question_type in {"entity", "numeric", "percentage"} else "general"
        _, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
            retrieval_mode=retrieval_mode,
        )
        has_context = len(context_parts) > 0
        if not has_context:
            yield self.llm_engine._no_answer_text(answer_language)
            return

        extracted = self.llm_engine._extract_answer_from_context(question, context)
        if extracted and question_type in {"numeric", "percentage", "entity"}:
            yield self.llm_engine.ensure_answer_language(
                question,
                extracted,
                answer_language=answer_language,
                context=context,
            )
            return

        for chunk in self.llm_engine.stream_answer(
            question,
            context,
            prompt_type=prompt_type,
            answer_language=answer_language,
        ):
            yield chunk

    def answer_without_rag(self, question: str, answer_language: str = "zh") -> Tuple[str, float]:
        start_time = time.time()
        answer = self.llm_engine.generate_answer_without_context(question, answer_language=answer_language)
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

