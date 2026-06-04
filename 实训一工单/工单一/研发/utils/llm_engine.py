# -*- coding: utf-8 -*-
"""LLM engine module with optional OpenAI-compatible API and local extraction."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Iterator, List

from src.config import Config
from src.prompts import build_prompt
from utils.language import detect_language, is_english
from utils.logger import get_logger
from utils.question_classifier import QuestionClassifier

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


logger = get_logger(__name__)


class LLMEngine:
    FALLBACK_MODELS = [
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
    ]

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.LLM_MODEL
        self.question_classifier = QuestionClassifier()
        self.client = self._init_client()

    def _init_client(self):
        if OpenAI is None or not Config.LLM_API_KEY:
            logger.info("openai disabled | reason=no_client_or_key")
            return None
        try:
            return OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_API_BASE_URL)
        except Exception:
            logger.exception("openai client init failed")
            return None

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text or "")
        text = re.sub(r"\b(?:rowspan|colspan|td|tr|table)\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\r\n，。；;")

    def _split_sentences(self, text: str) -> List[str]:
        pieces = re.split(r"(?<=[。！？；;?!.])|\n+", text or "")
        return [self._normalize_text(piece) for piece in pieces if self._normalize_text(piece)]

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9.]+|[\u4e00-\u9fff]+", (text or "").lower())
        stopwords = {
            "的",
            "了",
            "是",
            "在",
            "和",
            "与",
            "中",
            "公司",
            "武汉兴图新科电子股份有限公司",
            "报告期内",
            "分别",
            "多少",
            "哪个",
            "哪些",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "for",
            "with",
            "what",
            "which",
            "who",
            "how",
            "much",
            "many",
            "is",
            "are",
        }
        return [word for word in words if word not in stopwords and len(word) > 1][:12]

    def _expand_keywords(self, question: str) -> List[str]:
        keywords = self._extract_keywords(question)
        synonym_map = {
            "军用领域": ["国防领域", "军方客户", "军品业务", "军队用户", "直接和间接来自军方"],
            "军用": ["国防领域", "军方客户", "军品业务", "军队用户"],
            "民用领域": ["民用市场", "民品业务"],
            "收入": ["销售收入", "营业收入", "主营业务收入", "销售额"],
            "技术标准": ["技术规范", "全军第一个视频指挥系统技术标准", "某视频技术规范"],
            "参与制定": ["技术标准", "技术规范"],
            "募集资金": ["补充流动资金", "拟投入募集资金"],
            "revenue": ["收入", "销售收入", "营业收入", "销售额"],
            "income": ["收入", "销售收入", "营业收入", "销售额"],
        }
        expanded = list(keywords)
        lowered_question = (question or "").lower()
        for key, values in synonym_map.items():
            if key in (question or "") or key in lowered_question:
                expanded.extend(values)
        return list(dict.fromkeys(expanded))

    def understand_question(self, question: str, question_id: int = None) -> Dict:
        classified = self.question_classifier.classify(question, question_id=question_id)
        keywords = self._extract_keywords(question)
        return {
            "intent": classified["intent"],
            "prompt_type": classified["question_type"],
            "question_type": classified["question_type"],
            "matched_rule": classified.get("matched_rule", ""),
            "keywords": keywords,
            "is_ambiguous": len(keywords) <= 2,
            "sub_questions": [question] if question else [],
            "normalized_question": self._normalize_text(question),
            "language": classified.get("language", detect_language(question)),
        }

    def _score_sentence(self, sentence: str, keywords: List[str]) -> float:
        if not sentence:
            return 0.0
        score = 0.0
        lowered = sentence.lower()
        for keyword in keywords:
            if keyword.lower() in lowered:
                score += 1.0
        if any(ch.isdigit() for ch in sentence):
            score += 0.5
        if any(noise in sentence for noise in ["<td", "rowspan", "colspan"]):
            score -= 5.0
        return score + min(1.0, len(sentence) / 160.0)

    def _extract_money_values(self, text: str) -> List[str]:
        values = re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*万元|\d+(?:\.\d+)?\s*万元", text or "")
        return [re.sub(r"\s+", "", value) for value in values]

    def _extract_percent_values(self, text: str) -> List[str]:
        values = re.findall(r"\d+(?:\.\d+)?\s*%", text or "")
        return [re.sub(r"\s+", "", value) for value in values]

    def _extract_answer_from_context(self, question: str, context: str) -> str:
        if not context.strip():
            return ""

        clean_context = self._normalize_text(context)
        q = question or ""
        lowered_question = q.lower()

        if "参与制定" in q and ("技术标准" in q or "技术规范" in q):
            match = re.search(r"参与制定了?(全军第一个视频指挥系统技术标准（即《[^》]+》）)", clean_context)
            if match:
                return match.group(1)
            match = re.search(r"全军第一个视频指挥系统技术标准（即《[^》]+》）", clean_context)
            if match:
                return match.group(0)

        if "哪个领域" in q and "重要供应商" in q:
            match = re.search(r"已经成为([^，。；]+领域)的重要供应商", clean_context)
            if match:
                return match.group(1)

        if "国家科技进步一等奖" in q or "哪个工程" in q:
            match = re.search(r"“([^”]+工程)”[^。；]*荣获国家科技进步一等奖", clean_context)
            if match:
                return match.group(1)

        if "补充流动资金" in q:
            match = re.search(r"补充流动资金[^\d]{0,30}(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*万元)", clean_context)
            if match:
                return match.group(1).replace(" ", "")

        if "注册资本" in q:
            match = re.search(r"注册资本[^\d]{0,20}(\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*万元)", clean_context)
            if match:
                return match.group(1).replace(" ", "")

        if "法定代表人" in q:
            match = re.search(r"法定代表人[：:\s]*([\u4e00-\u9fff]{2,4})", clean_context)
            if match:
                return match.group(1)

        asks_revenue = any(key in q for key in ["收入", "销售额"]) or any(key in lowered_question for key in ["revenue", "income"])
        if asks_revenue:
            if any(key in q for key in ["军用领域", "军方", "国防", "军用"]):
                pattern = (
                    r"(?:直接和间接向国防客户的销售额合计|直接和间接来自军方[^，。；]*收入)"
                    r"分别为\s*([^。；%]+?)(?:，占|；|。)"
                )
                match = re.search(pattern, clean_context)
                if match:
                    values = self._extract_money_values(match.group(1))
                    if values:
                        return "、".join(values[:4])

            if any(key in q for key in ["民用领域", "民品", "民用"]):
                pattern = r"(?:民用市场实现销售收入|民品[^，。；]*收入)[^。；]*?分别为\s*([^。；%]+?万元(?:、\s*[^。；%]+?万元){1,6})"
                match = re.search(pattern, clean_context)
                if match:
                    values = self._extract_money_values(match.group(1))
                    if values:
                        return "、".join(values[:4])

            relevant_lines = []
            for line in re.split(r"[\n\r。；;]+", context):
                if any(key in line for key in ["直接和间接", "军方", "军用", "国防", "民用", "销售额", "销售收入"]):
                    relevant_lines.append(line.strip())
            money_text = " ".join(relevant_lines) if relevant_lines else clean_context
            if any(key in q for key in ["占比", "比重", "比例", "百分比"]):
                values = self._extract_percent_values(money_text)
            else:
                values = self._extract_money_values(money_text)
            if values:
                return "、".join(values[:4])

        keywords = self._expand_keywords(question)
        sentences = self._split_sentences(context)
        ranked = sorted(
            ((self._score_sentence(sentence, keywords), idx, sentence) for idx, sentence in enumerate(sentences)),
            key=lambda item: (-item[0], item[1]),
        )
        best_sentences = [sentence for score, _, sentence in ranked[:3] if score > 0]
        if best_sentences:
            answer = "；".join(best_sentences[:2])
            return self._normalize_text(answer)
        return ""

    def _needs_local_fallback(self, question: str, answer: str, prompt_type: str | None) -> bool:
        text = answer or ""
        if not text.strip():
            return True
        if any(noise in text for noise in ["<td", "</td", "rowspan", "colspan"]):
            return True
        if "暂时没有找到足够相关的信息" in text:
            return True
        if prompt_type in {"numeric", "percentage"} and not any(ch.isdigit() for ch in text):
            return True
        return False

    def _offline_answer(self, question: str, context: str = "") -> str:
        no_info = (
            "Based on the current knowledge base, no sufficiently relevant information was found."
            if is_english(question)
            else "根据当前知识库，暂时没有找到足够相关的信息。"
        )
        if not context.strip():
            return no_info
        extracted = self._extract_answer_from_context(question, context)
        return extracted or no_info

    def _build_prompt(self, question: str, context: str, prompt_type: str = None) -> str:
        if prompt_type is None:
            classified = self.question_classifier.classify(question)
            prompt_type = classified["question_type"]
        if prompt_type not in {"numeric", "entity"}:
            prompt_type = "general"
        return build_prompt(question=question, context=context, prompt_type=prompt_type, language=detect_language(question))

    def _chunk_text(self, text: str, chunk_size: int = 18) -> Iterator[str]:
        text = text or ""
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    def generate_answer(self, question: str, context: str, prompt_type: str = None) -> str:
        local_answer = self._extract_answer_from_context(question, context)
        if local_answer and self.question_classifier.classify(question)["question_type"] in {"numeric", "entity"}:
            return local_answer

        if self.client is None:
            logger.info("offline answer used | question=%s | context_len=%s", question, len(context or ""))
            return self._offline_answer(question, context)

        prompt = self._build_prompt(question, context, prompt_type=prompt_type)
        system_prompt = (
            "You are a professional bilingual PDF question-answering assistant."
            if is_english(question)
            else "你是一个专业的中文 PDF 问答助手。"
        )
        for model in [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                answer = self._normalize_text(resp.choices[0].message.content or "")
                if self._needs_local_fallback(question, answer, prompt_type) and local_answer:
                    logger.warning("llm answer rejected; local extraction used | question=%s", question)
                    return local_answer
                if answer:
                    self.model_name = model
                    return answer
            except Exception:
                logger.exception("generate_answer failed | question=%s | model=%s", question, model)
                continue
        return local_answer or self._offline_answer(question, context)

    def stream_answer(self, question: str, context: str, prompt_type: str = None) -> Iterable[str]:
        answer = self.generate_answer(question, context, prompt_type=prompt_type)
        for chunk in self._chunk_text(answer):
            yield chunk

    def generate_answer_without_context(self, question: str) -> str:
        if self.client is None:
            if is_english(question):
                return "Based on the current knowledge base, no sufficiently relevant information was found."
            return "根据当前知识库，暂时没有找到足够相关的信息。"
        try:
            system_prompt = (
                "You are a professional bilingual question-answering assistant. Answer in English."
                if is_english(question)
                else "你是一个专业的中文问答助手。"
            )
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
            )
            return self._normalize_text(resp.choices[0].message.content or "")
        except Exception:
            logger.exception("generate_answer_without_context failed | question=%s", question)
            if is_english(question):
                return "Based on the current knowledge base, no sufficiently relevant information was found."
            return "根据当前知识库，暂时没有找到足够相关的信息。"

    def query_intent(self, question: str, question_id: int = None) -> Dict:
        return self.understand_question(question, question_id=question_id)
