# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from src.config import Config
from src.constants import NEGATIVE_QUERY_FALLBACK
from src.models import RAGResponse
from src.utils.text_utils import _normalize_question_text
from src.validation.answer_validator import validate_answer_quality
from src.validation.negative_handler import handle_negative_query

class AnsweringMixin:
    def ask(
        self,
        question: str,
        question_id: int = None,
        retrieval_mode: str = "hybrid",
        session_id: str = "default",
        answer_language: str = "zh",
    ) -> RAGResponse:
        question = _normalize_question_text(question)
        answer_language = self._normalize_answer_language(answer_language)
        cache_key = self._answer_cache_key(question, question_id, retrieval_mode, session_id, answer_language)
        cached_response = self._get_cached_answer(cache_key)
        if cached_response is not None:
            return cached_response

        start_time = time.time()
        question_info = self.question_classifier.classify(question, question_id=question_id)
        question_type = question_info["question_type"]
        prompt_type = question_type if question_type in {"entity", "numeric", "percentage"} else "general"
        self.logger.info("ask start | question=%s | question_type=%s", question, question_type)

        multimodal_response = self._answer_multimodal_question(
            question=question,
            question_id=question_id,
            question_type=question_type,
            answer_language=answer_language,
            cache_key=cache_key,
            start_time=start_time,
        )
        if multimodal_response is not None:
            return multimodal_response

        if "未披露" in (question or ""):
            return self._negative_fallback_response(
                cache_key=cache_key,
                question=question,
                question_type=question_type,
                answer_language=answer_language,
                start_time=start_time,
                intent="undisclosed_precheck",
                context_label="undisclosed_precheck",
                reason="question_contains_未披露",
            )

        if self._is_undisclosed_related_party_question(question):
            return self._negative_fallback_response(
                cache_key=cache_key,
                question=question,
                question_type=question_type,
                answer_language=answer_language,
                start_time=start_time,
                intent="negative_query_precheck",
                context_label="negative_query_precheck",
                reason="undisclosed_or_absent_related_party",
            )

        negative_results = handle_negative_query(question, self.vector_store)
        if negative_results and negative_results != [NEGATIVE_QUERY_FALLBACK]:
            negative_answer = "、".join(negative_results)
            localized_answer = self._localize_answer(question, negative_answer, answer_language)
            source_chunk = {
                "content": "negative_query_handler",
                "metadata": {"results": negative_results},
                "relevance_score": 1.0 if negative_results != [NEGATIVE_QUERY_FALLBACK] else 0.0,
            }
            accuracy = 0.0 if negative_results == [NEGATIVE_QUERY_FALLBACK] else 1.0
            return self._cached_response(
                cache_key,
                question=question,
                answer=localized_answer,
                question_type=question_type,
                retrieval_mode="negative_query",
                retrieved_contexts=["negative_query_handler"],
                scores=[source_chunk["relevance_score"]],
                start_time=start_time,
                accuracy=accuracy,
                source_chunks=[source_chunk],
                has_context=negative_results != [NEGATIVE_QUERY_FALLBACK],
                query_analysis={
                    "intent": "negative_query",
                    "is_ambiguous": False,
                    "answer_language": answer_language,
                    "negative_query_results": negative_results,
                },
            )

        metadata_answer = self._metadata_answer(question)
        if metadata_answer:
            localized_answer = self._localize_answer(question, metadata_answer, answer_language)
            accuracy = self._estimate_accuracy(
                question=question,
                answer=localized_answer,
                retrieval_mode="metadata",
                has_context=True,
                scores=[1.0],
                retrieved_contexts=["document_metadata"],
            )
            return self._cached_response(
                cache_key,
                question=question,
                answer=localized_answer,
                question_type=question_type,
                retrieval_mode="metadata",
                retrieved_contexts=["document_metadata"],
                scores=[1.0],
                start_time=start_time,
                accuracy=accuracy,
                source_chunks=[self._metadata_source_chunk(question)],
                has_context=True,
                query_analysis={"intent": "metadata_lookup", "is_ambiguous": False, "answer_language": answer_language},
            )

        if negative_results == [NEGATIVE_QUERY_FALLBACK]:
            return self._negative_fallback_response(
                cache_key=cache_key,
                question=question,
                question_type=question_type,
                answer_language=answer_language,
                start_time=start_time,
                intent="negative_query",
                context_label="negative_query_handler",
                reason="negative_query_handler_fallback",
            )

        if self._is_liyuan_document(question):
            query_analysis, context, context_parts, scores, source_chunks = self._build_context(
                question,
                Config.TOP_K_RETRIEVAL,
                question_id=question_id,
                retrieval_mode=retrieval_mode,
            )
            table_answer = self._extract_liyuan_table_answer(question, context)
            if table_answer:
                localized_answer = self._localize_answer(question, table_answer, answer_language)
                accuracy = self._estimate_accuracy(
                    question=question,
                    answer=localized_answer,
                    retrieval_mode="table_metadata",
                    has_context=True,
                    scores=scores,
                    retrieved_contexts=context_parts,
                )
                return self._cached_response(
                    cache_key,
                    question=question,
                    answer=localized_answer,
                    question_type=question_type,
                    retrieval_mode="table_metadata",
                    retrieved_contexts=context_parts,
                    scores=scores,
                    start_time=start_time,
                    accuracy=accuracy,
                    source_chunks=source_chunks,
                    has_context=True,
                    query_analysis={
                        **query_analysis,
                        "intent": "table_metadata_lookup",
                        "is_ambiguous": False,
                        "answer_language": answer_language,
                    },
                )

        memory_match = None
        if Config.ENABLE_LONG_TERM_MEMORY_ANSWER and not self._should_skip_long_term_memory(question, question_type):
            memory_match = self.long_term_memory.search(question)
        if memory_match:
            localized_answer = self._localize_answer(question, memory_match["answer"], answer_language)
            memory_source_chunks = [
                {"content": memory_match["question"], "metadata": memory_match, "relevance_score": memory_match["score"]}
            ]
            memory_validation = validate_answer_quality(question, localized_answer, memory_source_chunks)
            if not memory_validation["is_valid"]:
                self.logger.warning(
                    "long-term memory rejected by quality validator | question=%s | reason=%s | confidence=%.3f",
                    question,
                    memory_validation["reason"],
                    memory_validation["confidence"],
                )
                memory_match = None
            else:
                accuracy = self._estimate_accuracy(
                    question=question,
                    answer=localized_answer,
                    retrieval_mode="long_term_memory",
                    has_context=True,
                    scores=[memory_match["score"]],
                    retrieved_contexts=[memory_match["question"]],
                )
                self.logger.info(
                    "long-term memory hit | question=%s | score=%.4f | elapsed=%.3f",
                    question,
                    memory_match["score"],
                    time.time() - start_time,
                )
                return self._cached_response(
                    cache_key,
                    question=question,
                    answer=localized_answer,
                    question_type=question_type,
                    memory_hit=True,
                    retrieval_mode="long_term_memory",
                    retrieved_contexts=[memory_match["question"]],
                    scores=[memory_match["score"]],
                    start_time=start_time,
                    accuracy=accuracy,
                    source_chunks=memory_source_chunks,
                    has_context=True,
                    query_analysis={
                        "intent": "long_term_memory",
                        "is_ambiguous": False,
                        "answer_language": answer_language,
                        "quality_validation": memory_validation,
                    },
                )
        query_analysis, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
            retrieval_mode=retrieval_mode,
        )
        has_context = len(context_parts) > 0
        if has_context:
            extracted = self._extract_complete_entity_answer(question, context) or self.llm_engine._extract_answer_from_context(
                question,
                context,
            )
            if extracted and question_type in {"numeric", "percentage", "entity"}:
                answer = self._localize_answer(
                    question,
                    self.llm_engine.postprocess_answer(question, extracted),
                    answer_language,
                )
            else:
                answer = self._localize_answer(
                    question,
                    self.llm_engine.postprocess_answer(
                        question,
                        self.llm_engine.generate_answer(
                            question,
                            context,
                            prompt_type=prompt_type,
                            answer_language=answer_language,
                        ),
                    ),
                    answer_language,
                )
        else:
            answer = self.llm_engine.no_answer_message(answer_language)

        validation = validate_answer_quality(question, answer, source_chunks)
        retried_for_quality = False
        if has_context and not validation["is_valid"]:
            self.logger.warning(
                "answer rejected by quality validator; retrying retrieval | question=%s | reason=%s | confidence=%.3f",
                question,
                validation["reason"],
                validation["confidence"],
            )
            (
                retry_analysis,
                retry_context,
                retry_context_parts,
                retry_scores,
                retry_source_chunks,
            ) = self._build_context_for_quality_retry(question, question_id=question_id)
            retry_has_context = len(retry_context_parts) > 0
            if retry_has_context:
                retry_extracted = self._extract_complete_entity_answer(
                    question,
                    retry_context,
                ) or self.llm_engine._extract_answer_from_context(question, retry_context)
                if retry_extracted and question_type in {"numeric", "percentage", "entity"}:
                    retry_answer = self._localize_answer(
                        question,
                        self.llm_engine.postprocess_answer(question, retry_extracted),
                        answer_language,
                    )
                else:
                    retry_answer = self._localize_answer(
                        question,
                        self.llm_engine.postprocess_answer(
                            question,
                            self.llm_engine.generate_answer(
                                question,
                                retry_context,
                                prompt_type=prompt_type,
                                answer_language=answer_language,
                            ),
                        ),
                        answer_language,
                    )
                retry_validation = validate_answer_quality(question, retry_answer, retry_source_chunks)
                retried_for_quality = True
                if retry_validation["is_valid"] or retry_validation["confidence"] > validation["confidence"]:
                    answer = retry_answer
                    validation = retry_validation
                    query_analysis = {**query_analysis, **retry_analysis}
                    context_parts = retry_context_parts
                    scores = retry_scores
                    source_chunks = retry_source_chunks
                    has_context = retry_has_context

        if not validation["is_valid"]:
            answer = self._no_answer_for_low_quality(validation, answer_language)

        response_time = time.time() - start_time
        accuracy = self._estimate_accuracy(
            question=question,
            answer=answer,
            retrieval_mode=retrieval_mode,
            has_context=has_context,
            scores=scores,
            retrieved_contexts=context_parts,
            validation=validation,
        )
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
        return self._cache_answer(
            cache_key,
            RAGResponse(
                question=question,
                answer=answer,
                question_type=question_type,
                memory_hit=False,
                retrieval_mode=retrieval_mode,
                retrieved_contexts=context_parts,
                scores=scores,
                response_time=response_time,
                accuracy=accuracy,
                source_chunks=source_chunks,
                has_context=has_context,
                query_analysis={
                    **query_analysis,
                    "answer_language": answer_language,
                    "quality_validation": validation,
                    "quality_retry_used": retried_for_quality,
                },
            ),
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
        question = _normalize_question_text(question)
        answer_language = self._normalize_answer_language(answer_language)
        question_info = self.question_classifier.classify(question, question_id=question_id)
        question_type = question_info["question_type"]
        prompt_type = question_type if question_type in {"entity", "numeric", "percentage"} else "general"
        metadata_answer = self._metadata_answer(question)
        if metadata_answer:
            yield self._localize_answer(question, metadata_answer, answer_language)
            return
        _, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
            retrieval_mode=retrieval_mode,
        )
        has_context = len(context_parts) > 0
        if not has_context:
            yield self.llm_engine.no_answer_message(answer_language)
            return

        extracted = self._extract_complete_entity_answer(question, context) or self.llm_engine._extract_answer_from_context(
            question,
            context,
        )
        if extracted and question_type in {"numeric", "percentage", "entity"}:
            yield self._localize_answer(question, self.llm_engine.postprocess_answer(question, extracted), answer_language)
            return

        for chunk in self.llm_engine.stream_answer(
            question,
            context,
            prompt_type=prompt_type,
            answer_language=answer_language,
        ):
            yield chunk

    def answer_without_rag(self, question: str, answer_language: str = "zh") -> Tuple[str, float]:
        question = _normalize_question_text(question)
        start_time = time.time()
        answer = self.llm_engine.generate_answer_without_context(question, answer_language=answer_language)
        response_time = time.time() - start_time
        self.logger.info("answer_without_rag done | question=%s | elapsed=%.3f", question, response_time)
        return answer, response_time

    def record_feedback(self, question: str, answer: str, helpful: bool, question_type: str = "entity") -> None:
        self.logger.info("feedback recorded | helpful=%s | question=%s", helpful, question)

    def batch_answer(self, questions: List[Dict], answer_language: str = "zh") -> List[Dict]:
        results = []
        for q in questions:
            response = self.ask(q["question"], question_id=q.get("id"), answer_language=answer_language)
            answer_no_rag, time_no_rag = self.answer_without_rag(q["question"], answer_language=answer_language)
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "rag_answer": response.answer,
                    "question_type": response.question_type,
                    "retrieval_mode": response.retrieval_mode,
                    "rag_response_time": response.response_time,
                    "accuracy": response.accuracy,
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
