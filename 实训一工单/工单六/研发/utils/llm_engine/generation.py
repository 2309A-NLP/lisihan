# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Iterator, List

from src.config import Config
from src.prompts import build_prompt
from utils.logger import get_logger


logger = get_logger(__name__)


class GenerationMixin:
    def _build_prompt(self, question: str, context: str, prompt_type: str = None, answer_language: str = "zh") -> str:
        if prompt_type is None:
            classified = self.question_classifier.classify(question)
            prompt_type = classified["question_type"]
        if prompt_type == "percentage":
            prompt_type = "percentage"
        elif prompt_type == "numeric":
            prompt_type = "numeric"
        elif prompt_type != "entity":
            prompt_type = "general"
        return build_prompt(
            question=question,
            context=self._limit_context(context),
            prompt_type=prompt_type,
            answer_language=self._normalize_answer_language(answer_language),
        )

    def _limit_context(self, context: str) -> str:
        max_chars = getattr(Config, "LLM_CONTEXT_MAX_CHARS", 6000)
        if len(context or "") <= max_chars:
            return context or ""

        parts = re.split(r"\n\n---\n\n", context or "")
        selected: List[str] = []
        total = 0
        for part in parts:
            if not part.strip():
                continue
            part_len = len(part)
            if selected and total + part_len + 7 > max_chars:
                break
            selected.append(part)
            total += part_len + 7
        if selected:
            return "\n\n---\n\n".join(selected)
        return (context or "")[:max_chars]

    def _chunk_text(self, text: str, chunk_size: int = 18) -> Iterator[str]:
        text = text or ""
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    def generate_answer(self, question: str, context: str, prompt_type: str = None, answer_language: str = "zh") -> str:
        language = self._normalize_answer_language(answer_language)
        if self.client is None:
            logger.info("offline answer used | question=%s | context_len=%s", question, len(context or ""))
            return self._offline_answer(question, context, answer_language=language)

        prompt = self._build_prompt(question, context, prompt_type=prompt_type, answer_language=language)
        system_prompt = (
            "You are a professional bilingual question-answering assistant. Answer in English when requested."
            if language == "en"
            else "你是一个专业的中文问答助手。"
        )
        for model in [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=Config.LLM_MAX_TOKENS,
                )
                answer = self._normalize_text(resp.choices[0].message.content or "")
                answer = self.postprocess_answer(question, answer)
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                if finish_reason == "length" or self._needs_local_fallback(question, answer, prompt_type):
                    fallback = self._extract_answer_from_context(question, context)
                    if fallback:
                        logger.warning("llm returned no-answer despite context; local extraction used | question=%s", question)
                        return self.localize_answer(question, fallback, language)
                if answer:
                    self.model_name = model
                    return self.localize_answer(question, self.postprocess_answer(question, answer), language)
            except Exception:
                logger.exception("generate_answer failed | question=%s | model=%s", question, model)
                continue
        return self._offline_answer(question, context, answer_language=language)

    def stream_answer(self, question: str, context: str, prompt_type: str = None, answer_language: str = "zh") -> Iterable[str]:
        language = self._normalize_answer_language(answer_language)
        if self.client is None:
            answer = self._offline_answer(question, context, answer_language=language)
            for chunk in self._chunk_text(answer):
                yield chunk
            return

        prompt = self._build_prompt(question, context, prompt_type=prompt_type, answer_language=language)
        system_prompt = (
            "You are a professional bilingual question-answering assistant. Answer in English when requested."
            if language == "en"
            else "你是一个专业的中文问答助手。"
        )
        for model in [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]:
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=Config.LLM_MAX_TOKENS,
                    stream=True,
                )
                any_chunk = False
                for event in stream:
                    delta = event.choices[0].delta.content or ""
                    if delta:
                        any_chunk = True
                        yield delta
                if any_chunk:
                    self.model_name = model
                    return
            except Exception:
                logger.exception("stream_answer failed | question=%s | model=%s", question, model)
                continue
        fallback = self._offline_answer(question, context, answer_language=language)
        for chunk in self._chunk_text(fallback):
            yield chunk

    def generate_answer_without_context(self, question: str, answer_language: str = "zh") -> str:
        language = self._normalize_answer_language(answer_language)
        if self.client is None:
            logger.info("offline answer used without context | question=%s", question)
            return self.no_answer_message(language)
        try:
            user_prompt = question if language == "zh" else f"Answer the following question in English:\n{question}"
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a professional bilingual question-answering assistant."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
            answer = resp.choices[0].message.content or ""
            return self.localize_answer(question, self.postprocess_answer(question, answer), language)
        except Exception:
            logger.exception("generate_answer_without_context failed | question=%s", question)
            return self.no_answer_message(language)

    def query_intent(self, question: str, question_id: int = None) -> Dict:
        return self.understand_question(question, question_id=question_id)
