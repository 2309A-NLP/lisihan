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
    def _is_number_required_question(self, question: str, prompt_type: str | None = None) -> bool:
        q = question or ""
        if prompt_type in {"numeric", "percentage"}:
            return True
        return any(term in q for term in ["多少", "金额", "同比", "增长", "信用减值损失", "比例", "占比", "比重"])

    def _focused_context_for_generation(self, question: str, context: str) -> str:
        if not context:
            return ""
        q = question or ""
        focus_groups = [
            ["证券分公司", "分公司", "名称", "地址", "分支机构"],
            ["信用减值损失", "同比", "增长", "亿元", "万元", "%", "％"],
            ["哑铃型", "资产配置", "长期利率债", "权益类资产"],
            ["组织架构", "调整", "委员会", "机构与交易业务委员会", "研究与机构业务委员会", "交易投资业务委员会"],
        ]
        active_terms: List[str] = []
        for group in focus_groups:
            if any(term in q for term in group[:2]):
                active_terms.extend(group)
        if not active_terms:
            return context

        sentences = self._split_sentences(context)
        ranked = []
        for index, sentence in enumerate(sentences):
            score = sum(2 for term in active_terms if term in sentence)
            if any(ch.isdigit() for ch in sentence):
                score += 1
            if score:
                ranked.append((score, index, sentence))
        if not ranked:
            return context
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = [self._normalize_text(sentence) for _, _, sentence in ranked[:8]]
        return "\n".join(selected)

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
        focused_context = self._focused_context_for_generation(question, context)
        prompt = build_prompt(
            question=question,
            context=self._limit_context(focused_context),
            prompt_type=prompt_type,
            answer_language=self._normalize_answer_language(answer_language),
        )
        q = question or ""
        extra_rules = []
        if self._is_number_required_question(question, prompt_type):
            extra_rules.append("数值类问题必须从资料中抽取具体数字、金额、百分比或同比变化；如果资料中有数字，禁止只回答概括性描述。")
        if "信用减值损失" in q:
            extra_rules.append("本题必须同时查找“信用减值损失”的金额和“同比增长/同比增加”的百分比，答案格式优先为“信用减值损失为X亿元，同比增长Y%”。")
        if "分公司" in q:
            extra_rules.append("本题只回答证券分公司的数量和名称/地点列表，忽略债券、募集资金、承销等无关内容。")
        if "哑铃型" in q or "资产配置策略" in q:
            extra_rules.append("本题只回答“哑铃型”资产配置策略的具体做法，优先保留长期利率债、权益类资产等关键词。")
        if "组织架构" in q:
            extra_rules.append("本题必须回答具体组织架构调整内容，优先保留被合并/调整的委员会名称和调整后的委员会名称。")
        if extra_rules:
            prompt += "\n\n补充硬性要求：\n" + "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(extra_rules, 1))
        return prompt

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

        focused_local = self._extract_answer_from_context(question, context)
        if focused_local and (
            self._is_number_required_question(question, prompt_type)
            or any(term in (question or "") for term in ["分公司", "哑铃型", "资产配置策略", "组织架构"])
        ):
            logger.info("focused local answer used before llm | question=%s", question)
            return self.localize_answer(question, self.postprocess_answer(question, focused_local), language)

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
