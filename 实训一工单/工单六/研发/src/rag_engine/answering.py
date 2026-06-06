# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件属于 PDF 招股说明书智能问答系统，用于在 RAGEngine.ask 中接入工单五
多轮对话与指代消解，并保留工单一到工单四的原有问答能力。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Tuple

from src.config import Config
from src.constants import NEGATIVE_QUERY_FALLBACK
from src.models import RAGResponse
from src.utils.text_utils import _normalize_question_text
from src.validation.answer_validator import validate_answer_quality
from src.validation.negative_handler import handle_negative_query


class AnsweringMixin:
    def _generate_answer_from_context(
        self,
        *,
        question: str,
        context: str,
        has_context: bool,
        question_type: str,
        prompt_type: str,
        answer_language: str,
    ) -> str:
        if not has_context:
            return self.llm_engine.no_answer_message(answer_language)

        extracted = self._extract_complete_entity_answer(question, context) or self.llm_engine._extract_answer_from_context(
            question,
            context,
        )
        if extracted and question_type in {"numeric", "percentage", "entity"}:
            return self._localize_answer(
                question,
                self.llm_engine.postprocess_answer(question, extracted),
                answer_language,
            )
        if extracted and getattr(Config, "FAST_LOCAL_EXTRACTION", True) and answer_language == "zh":
            return self._localize_answer(
                question,
                self.llm_engine.postprocess_answer(question, extracted),
                answer_language,
            )

        generated = self.llm_engine.generate_answer(
            question,
            context,
            prompt_type=prompt_type,
            answer_language=answer_language,
        )
        return self._localize_answer(
            question,
            self.llm_engine.postprocess_answer(question, generated),
            answer_language,
        )

    def _retry_answer_for_quality(
        self,
        *,
        question: str,
        question_id: int,
        question_type: str,
        prompt_type: str,
        answer_language: str,
        answer: str,
        validation: Dict,
        query_analysis: Dict,
        context_parts: List[str],
        scores: List[float],
        source_chunks: List[Dict],
        has_context: bool,
        start_time: float | None = None,
    ):
        retried_for_quality = False
        if not has_context or validation["is_valid"]:
            return answer, validation, query_analysis, context_parts, scores, source_chunks, has_context, retried_for_quality
        if start_time is not None and time.time() - start_time >= getattr(Config, "QUALITY_RETRY_MAX_ELAPSED_SECONDS", 2.2):
            self.logger.warning(
                "quality retry skipped by time budget | question=%s | elapsed=%.3f",
                question,
                time.time() - start_time,
            )
            return answer, validation, query_analysis, context_parts, scores, source_chunks, has_context, retried_for_quality

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
            retry_answer = self._generate_answer_from_context(
                question=question,
                context=retry_context,
                has_context=retry_has_context,
                question_type=question_type,
                prompt_type=prompt_type,
                answer_language=answer_language,
            )
            retry_validation = validate_answer_quality(question, retry_answer, retry_source_chunks)
            retried_for_quality = True
            if retry_validation["is_valid"] or retry_validation["confidence"] > validation["confidence"]:
                return (
                    retry_answer,
                    retry_validation,
                    {**query_analysis, **retry_analysis},
                    retry_context_parts,
                    retry_scores,
                    retry_source_chunks,
                    retry_has_context,
                    retried_for_quality,
                )

        return answer, validation, query_analysis, context_parts, scores, source_chunks, has_context, retried_for_quality

    def _resolve_question_for_session(self, question: str, session_id: str):
        history = []
        if hasattr(self, "session_manager"):
            history = self.session_manager.get_history(session_id=session_id)
        if hasattr(self, "coreference_resolver"):
            return self.coreference_resolver.resolve(question, history)
        return None

    def _finalize_session_response(self, response: RAGResponse, *, original_question: str, session_id: str, coreference):
        if coreference is None:
            return response

        coreference_payload = coreference.to_dict()
        query_analysis = {
            **(response.query_analysis or {}),
            "session_id": session_id,
            "original_question": original_question,
            "resolved_question": coreference.resolved_question,
            "coreference": coreference_payload,
        }
        finalized = replace(response, query_analysis=query_analysis)

        if hasattr(self, "session_manager"):
            self.session_manager.add_turn(
                session_id=session_id,
                question=original_question,
                resolved_question=coreference.resolved_question,
                answer=finalized.answer,
                mentioned_companies=coreference.mentioned_companies or [],
                current_company=coreference.current_company,
                metadata={
                    "question_type": finalized.question_type,
                    "retrieval_mode": finalized.retrieval_mode,
                    "has_context": finalized.has_context,
                    "coreference_resolved": coreference.is_resolved,
                },
            )
        return finalized

    def _answer_liyuan_table_metadata(
        self,
        *,
        question: str,
        question_id: int | None,
        question_type: str,
        retrieval_mode: str,
        retrieval_config: Dict | None,
        answer_language: str,
        cache_key: str,
        start_time: float,
    ) -> RAGResponse | None:
        if not self._is_liyuan_document(question):
            return None

        query_analysis, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
            retrieval_mode=retrieval_mode,
            retrieval_config=retrieval_config,
        )
        table_answer = self._extract_liyuan_table_answer(question, context)
        if not table_answer and any(term in (question or "") for term in ["募集资金拟投资", "募集资金用途", "投资哪些项目"]):
            table_answer = self.llm_engine._extract_answer_from_context(question, context)
        if not table_answer:
            return None

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

    def _answer_from_long_term_memory(
        self,
        *,
        question: str,
        question_type: str,
        answer_language: str,
        cache_key: str,
        start_time: float,
    ) -> RAGResponse | None:
        if not getattr(Config, "ENABLE_LONG_TERM_MEMORY_ANSWER", False):
            return None
        if self._should_skip_long_term_memory(question, question_type):
            return None

        memory_match = self.long_term_memory.search(question)
        if not memory_match:
            return None

        localized_answer = self._localize_answer(question, memory_match["answer"], answer_language)
        memory_source_chunks = [
            {
                "content": memory_match["question"],
                "metadata": memory_match,
                "relevance_score": memory_match["score"],
            }
        ]
        memory_validation = validate_answer_quality(question, localized_answer, memory_source_chunks)
        if not memory_validation["is_valid"]:
            self.logger.warning(
                "long-term memory rejected by quality validator | question=%s | reason=%s | confidence=%.3f",
                question,
                memory_validation["reason"],
                memory_validation["confidence"],
            )
            return None

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

    def ask(
        self,
        question: str,
        question_id: int = None,
        retrieval_mode: str = "hybrid",
        session_id: str = "default",
        answer_language: str = "zh",
        retrieval_config: Dict | None = None,
    ) -> RAGResponse:
        session_id = session_id or "default"
        original_question = _normalize_question_text(question)
        answer_language = self._normalize_answer_language(answer_language)
        coreference = self._resolve_question_for_session(original_question, session_id)
        question = coreference.resolved_question if coreference is not None else original_question
        cache_key = self._answer_cache_key(
            question,
            question_id,
            retrieval_mode,
            session_id,
            answer_language,
            retrieval_config=retrieval_config,
        )
        cached_response = self._get_cached_answer(cache_key)
        if cached_response is not None:
            return self._finalize_session_response(
                cached_response,
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

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
            return self._finalize_session_response(
                multimodal_response,
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

        if "未披露" in (question or ""):
            return self._finalize_session_response(
                self._negative_fallback_response(
                    cache_key=cache_key,
                    question=question,
                    question_type=question_type,
                    answer_language=answer_language,
                    start_time=start_time,
                    intent="undisclosed_precheck",
                    context_label="undisclosed_precheck",
                    reason="question_contains_未披露",
                ),
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

        if self._is_undisclosed_related_party_question(question):
            return self._finalize_session_response(
                self._negative_fallback_response(
                    cache_key=cache_key,
                    question=question,
                    question_type=question_type,
                    answer_language=answer_language,
                    start_time=start_time,
                    intent="negative_query_precheck",
                    context_label="negative_query_precheck",
                    reason="undisclosed_or_absent_related_party",
                ),
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
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
            return self._finalize_session_response(
                self._cached_response(
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
                ),
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
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
            return self._finalize_session_response(
                self._cached_response(
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
                ),
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

        if negative_results == [NEGATIVE_QUERY_FALLBACK]:
            return self._finalize_session_response(
                self._negative_fallback_response(
                    cache_key=cache_key,
                    question=question,
                    question_type=question_type,
                    answer_language=answer_language,
                    start_time=start_time,
                    intent="negative_query",
                    context_label="negative_query_handler",
                    reason="negative_query_handler_fallback",
                ),
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

        table_response = self._answer_liyuan_table_metadata(
            question=question,
            question_id=question_id,
            question_type=question_type,
            retrieval_mode=retrieval_mode,
            retrieval_config=retrieval_config,
            answer_language=answer_language,
            cache_key=cache_key,
            start_time=start_time,
        )
        if table_response is not None:
            return self._finalize_session_response(
                table_response,
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

        memory_response = self._answer_from_long_term_memory(
            question=question,
            question_type=question_type,
            answer_language=answer_language,
            cache_key=cache_key,
            start_time=start_time,
        )
        if memory_response is not None:
            return self._finalize_session_response(
                memory_response,
                original_question=original_question,
                session_id=session_id,
                coreference=coreference,
            )

        query_analysis, context, context_parts, scores, source_chunks = self._build_context(
            question,
            Config.TOP_K_RETRIEVAL,
            question_id=question_id,
            retrieval_mode=retrieval_mode,
            retrieval_config=retrieval_config,
        )
        has_context = len(context_parts) > 0
        answer = self._generate_answer_from_context(
            question=question,
            context=context,
            has_context=has_context,
            question_type=question_type,
            prompt_type=prompt_type,
            answer_language=answer_language,
        )

        validation = validate_answer_quality(question, answer, source_chunks)
        (
            answer,
            validation,
            query_analysis,
            context_parts,
            scores,
            source_chunks,
            has_context,
            retried_for_quality,
        ) = self._retry_answer_for_quality(
            question=question,
            question_id=question_id,
            question_type=question_type,
            prompt_type=prompt_type,
            answer_language=answer_language,
            answer=answer,
            validation=validation,
            query_analysis=query_analysis,
            context_parts=context_parts,
            scores=scores,
            source_chunks=source_chunks,
            has_context=has_context,
            start_time=start_time,
        )

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
        return self._finalize_session_response(
            self._cache_answer(
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
                        "retrieval_config": retrieval_config or {},
                        "quality_validation": validation,
                        "quality_retry_used": retried_for_quality,
                    },
                ),
            ),
            original_question=original_question,
            session_id=session_id,
            coreference=coreference,
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
        retrieval_config: Dict | None = None,
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
            retrieval_config=retrieval_config,
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

    def record_feedback(
        self,
        question: str,
        answer: str,
        helpful: bool,
        question_type: str = "entity",
        source_chunks: List[Dict] = None,
    ) -> None:
        self.logger.info("feedback recorded | helpful=%s | question=%s", helpful, question)
        if hasattr(self, "vector_store") and hasattr(self.vector_store, "record_feedback"):
            if source_chunks is not None:
                self.vector_store.record_feedback(source_chunks, helpful)
                return
            cached_entries = list(getattr(self, "_answer_cache", {}).values())
            for response, _ in cached_entries:
                if response.question == question and response.answer == answer:
                    self.vector_store.record_feedback(response.source_chunks, helpful)
                    break

    def batch_answer(
        self,
        questions: List[Dict],
        answer_language: str = "zh",
        retrieval_mode: str = "hybrid",
        retrieval_config: Dict | None = None,
    ) -> List[Dict]:
        results = []
        for q in questions:
            response = self.ask(
                q["question"],
                question_id=q.get("id"),
                retrieval_mode=retrieval_mode,
                answer_language=answer_language,
                retrieval_config=retrieval_config,
            )
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
