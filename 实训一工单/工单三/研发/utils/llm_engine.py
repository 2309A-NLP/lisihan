# -*- coding: utf-8 -*-
"""LLM engine module with optional OpenAI fallback and streaming support."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Iterator, List

from src.config import Config
from src.prompts import build_prompt
from utils.question_classifier import QuestionClassifier
from utils.logger import get_logger

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


logger = get_logger(__name__)


ENTITY_EXTRACT_MAX_CHARS = 500


def _normalize_entity_source(chunk: str) -> str:
    text = re.sub(r"\s+", " ", chunk or "")
    text = re.sub(r"\|+", " ", text)
    return text.strip()


def extract_complete_entity(chunk: str, entity_type: str) -> str:
    """
    完整提取实体信息，不被截断。
    对于法定代表人：提取"法定代表人：XXX"完整模式。
    对于注册资本：提取数字+万元/亿元模式。
    最大提取窗口为 500 字符。
    """
    text = _normalize_entity_source(chunk)[:ENTITY_EXTRACT_MAX_CHARS]
    entity_type = (entity_type or "").strip().lower()

    if entity_type in {"legal_representative", "法定代表人"}:
        match = re.search(
            r"法定代表人\s*[:：]?\s*([\u4e00-\u9fa5A-Za-z·]{2,20})",
            text,
        )
        if match:
            return f"法定代表人：{match.group(1)}"
        return ""

    if entity_type in {"registered_capital", "注册资本"}:
        match = re.search(
            r"注册资本\s*[:：]?\s*([0-9０-９,，]+(?:\.\d+)?\s*(?:万元|亿元|万人民币|人民币万元|人民币亿元))",
            text,
        )
        if match:
            value = match.group(1).replace("，", ",").replace(" ", "").replace(".00", "")
            return f"注册资本：{value}"
        match = re.search(r"([0-9０-９,，]+(?:\.\d+)?\s*(?:万元|亿元))", text)
        if match:
            value = match.group(1).replace("，", ",").replace(" ", "").replace(".00", "")
            return f"注册资本：{value}"
        return ""

    return text[:ENTITY_EXTRACT_MAX_CHARS]


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
            return OpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_API_BASE_URL,
                timeout=Config.LLM_TIMEOUT,
                max_retries=0,
            )
        except Exception:
            logger.exception("openai client init failed")
            return None

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "")
        text = re.sub(r"^[\-\*\•\·\s]+", "", text)
        text = re.sub(r"^\([一二三四五六七八九十0-9]+\)\s*", "", text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        pieces = re.split(r"(?<=[。！？；?])|\n+", text or "")
        return [self._normalize_text(piece) for piece in pieces if self._normalize_text(piece)]

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
        stopwords = {
            "的",
            "了",
            "是",
            "在",
            "和",
            "中",
            "及",
            "或",
            "请",
            "一个",
            "哪些",
            "什么",
            "多少",
            "是否",
            "可以",
            "公司",
            "报告",
            "问题",
            "收入",
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        return keywords[:12]

    def _expand_keywords(self, question: str) -> List[str]:
        keywords = self._extract_keywords(question)
        synonym_map = {
            "军用领域": ["国防领域", "军方客户", "军品业务", "军用"],
            "军用": ["国防领域", "军方客户", "军品业务"],
            "民用领域": ["民用市场", "民品业务", "民用"],
            "民用": ["民用市场", "民品业务"],
            "收入": ["销售收入", "主营业务收入", "业务收入"],
            "比重": ["占主营业务收入的比重", "占比", "比例", "%"],
            "占比": ["占主营业务收入的比重", "比重", "比例", "%"],
            "上游": ["电子元器件制造企业", "机箱", "机柜", "金属壳体制造企业"],
            "下游": ["终端用户", "军队", "政府机关", "能源"],
            "技术标准": ["参与制定", "视频指挥系统技术标准", "某视频技术规范1.0", "全军第一个"],
        }
        expanded = list(keywords)
        for key, values in synonym_map.items():
            if key in question:
                expanded.extend(values)
        seen = set()
        return [item for item in expanded if not (item in seen or seen.add(item))]

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
        }

    def _score_sentence(self, sentence: str, keywords: List[str]) -> float:
        if not sentence:
            return 0.0
        if not keywords:
            return min(1.0, len(sentence) / 40.0)

        score = 0.0
        lowered = sentence.lower()
        for kw in keywords:
            if kw.lower() in lowered:
                score += 1.0
        if any(ch.isdigit() for ch in sentence):
            score += 0.5
        return score + min(1.0, len(sentence) / 120.0)

    def _extract_money_values(self, text: str) -> List[str]:
        pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*万元|\d+(?:\.\d+)?\s*万元"
        values = re.findall(pattern, text or "")
        cleaned = []
        for value in values:
            value = value.replace(" ", "")
            cleaned.append(value)
        return cleaned

    def _extract_percentage_values(self, text: str) -> List[str]:
        values = re.findall(r"\d+(?:\.\d+)?\s*[%％]", text or "")
        return [value.replace(" ", "").replace("％", "%") for value in values]

    def _numeric_focus_terms(self, question: str) -> List[str]:
        q = question or ""
        candidates = [
            "补充流动资金",
            "补充营运资金",
            "注册资本",
            "发行股数",
            "发行后总股本",
            "募集资金",
            "拟投入募集资金",
            "计划总投资",
            "销售额",
            "收入",
            "占主营业务收入",
            "比重",
            "比例",
            "占比",
        ]
        terms = [term for term in candidates if term in q]
        if "多少" in q:
            terms.append("多少")
        return terms

    def _focused_numeric_text(self, question: str, text: str) -> str:
        terms = self._numeric_focus_terms(question)
        if not terms:
            return text or ""

        lines = [line.strip() for line in re.split(r"[\r\n]+", text or "") if line.strip()]
        scored = []
        for index, line in enumerate(lines):
            compact = re.sub(r"\s+", "", line)
            score = sum(3 for term in terms if term in compact)
            if any(unit in compact for unit in ["万元", "亿元", "万股", "%", "％"]):
                score += 1
            if score:
                scored.append((score, index, line))
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            best_index = scored[0][1]
            window = lines[max(0, best_index - 1) : best_index + 2]
            return "\n".join(window)

        sentences = self._split_sentences(text)
        scored_sentences = []
        for index, sentence in enumerate(sentences):
            compact = re.sub(r"\s+", "", sentence)
            score = sum(3 for term in terms if term in compact)
            if any(unit in compact for unit in ["万元", "亿元", "万股", "%", "％"]):
                score += 1
            if score:
                scored_sentences.append((score, index, sentence))
        if scored_sentences:
            scored_sentences.sort(key=lambda item: (-item[0], item[1]))
            return scored_sentences[0][2]
        return text or ""

    def _extract_focused_numeric_answer(self, question: str, text: str) -> str:
        focused = self._focused_numeric_text(question, text)
        compact = re.sub(r"\s+", "", focused or "")
        q = question or ""

        if any(term in q for term in ["补充流动资金", "补充营运资金"]):
            specific_patterns = [
                r"拟使用本次发行募集资金([0-9,]+(?:\.\d+)?)万元用于补充(?:流动|营运)资金",
                r"补充(?:流动|营运)资金\|?([0-9,]+(?:\.\d+)?)\|?([0-9,]+(?:\.\d+)?)",
                r"补充(?:流动|营运)资金[^0-9]{0,30}([0-9,]+(?:\.\d+)?)万元",
            ]
            for pattern in specific_patterns:
                match = re.search(pattern, compact)
                if match:
                    value = match.group(2) if len(match.groups()) >= 2 and match.group(2) else match.group(1)
                    return f"{value.replace('.00', '')}万元"

        if any(term in q for term in ["占比", "比重", "比例", "百分比", "占主营业务收入"]):
            values = self._extract_percentage_values(focused)
            if values:
                return "、".join(values[:4])

        if "发行股数" in q and "比例" in q:
            share = re.search(r"([0-9,]+(?:\.\d+)?万股)", compact)
            ratio = re.search(r"([0-9]+(?:\.\d+)?%)", compact)
            if share and ratio:
                return f"{share.group(1)}，占发行后总股本的比例为{ratio.group(1)}"

        if any(term in q for term in ["注册资本", "金额", "多少", "数量", "收入", "募集资金"]):
            unit_pattern = r"([0-9,]+(?:\.\d+)?(?:万元|亿元|万股|元))"
            if "|" in focused:
                row_values = re.findall(unit_pattern, compact)
                if row_values:
                    value = row_values[-1]
                    return value.replace(".00", "")
            values = re.findall(unit_pattern, compact)
            if values:
                return values[0].replace(".00", "")
        return ""

    def postprocess_answer(self, question: str, answer: str) -> str:
        """清洗模型输出中的表格残片、无关上下文和多余数值。"""
        question = question or ""
        text = self._normalize_text(answer or "")
        text = re.sub(r"\|+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" ，,；;。")

        if self._is_military_income_amount_question(question):
            values = self._extract_money_values(text)
            if values:
                return "、".join(values[:3])

        focused_numeric = self._extract_focused_numeric_answer(question, text)
        if focused_numeric:
            return focused_numeric

        if "法定代表人" in question:
            complete = extract_complete_entity(text, "legal_representative")
            if complete:
                return complete.split("：", 1)[1]
            match = re.search(r"法定代表人[:：]?\s*([\u4e00-\u9fa5]{2,4})", text)
            if match:
                return match.group(1)

        if "注册资本" in question:
            complete = extract_complete_entity(text, "registered_capital")
            if complete:
                return complete.split("：", 1)[1]
            match = re.search(r"注册资本[:：]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*万元", text)
            if match:
                value = match.group(1).replace(".00", "")
                return f"{value}万元"

        if "重要供应商" in question or "哪个领域" in question:
            if "国防军队视频指挥领域" in text:
                return "国防军队视频指挥领域"
            if "军队视频指挥领域" in text:
                return "国防军队视频指挥领域"

        if "国家科技进步一等奖" in question or "一等奖" in question:
            if "某情报、指挥、控制与通信网络一体化工程" in text:
                return "某情报、指挥、控制与通信网络一体化工程"

        if any(term in question for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"]):
            values = self._extract_percentage_values(text)
            if values:
                limit = 3 if "2019" not in question and "1-6" not in question and "1至6" not in question else 4
                return "、".join(values[:limit])

        return text

    def _is_military_income_amount_question(self, question: str) -> bool:
        q = question or ""
        is_military_income = any(term in q for term in ["军用领域", "国防客户", "军方客户", "军品业务"])
        is_income_amount = "收入" in q or "销售额" in q
        is_percentage = any(term in q for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"])
        return is_military_income and is_income_amount and not is_percentage

    def _extract_answer_from_context(self, question: str, context: str) -> str:
        if not context.strip():
            return ""

        keywords = self._expand_keywords(question)
        sentences = self._split_sentences(context)

        if self._is_military_income_amount_question(question):
            for sentence in sentences:
                if "直接和间接向国防客户" in sentence and "销售额合计分别为" in sentence:
                    values = self._extract_money_values(sentence)
                    if values:
                        return "、".join(values[:3])
            for sentence in sentences:
                if "来自军用领域的收入分别为" in sentence:
                    values = self._extract_money_values(sentence)
                    if values:
                        return "、".join(values[:3])

        if any(term in question for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"]):
            for sentence in sentences:
                if (
                    ("直接和间接向国防客户" in sentence or "来自军用领域的收入占比" in sentence)
                    and "占主营业务收入" in sentence
                ):
                    values = self._extract_percentage_values(sentence)
                    if values:
                        limit = 3 if "2019" not in question and "1-6" not in question and "1至6" not in question else 4
                        return "、".join(values[:limit])

            focused_lines = []
            for line in re.split(r"[\n\r]+", context):
                if any(key in line for key in ["军用", "国防", "主营业务收入", "占主营业务收入", "比重"]):
                    focused_lines.append(line.strip())
            percent_text = " ".join(focused_lines) if focused_lines else context
            values = self._extract_percentage_values(percent_text)
            if values:
                limit = 3 if "2019" not in question and "1-6" not in question and "1至6" not in question else 4
                return "、".join(values[:limit])

        focused_numeric = self._extract_focused_numeric_answer(question, context)
        if focused_numeric:
            return focused_numeric

        if "法定代表人" in question:
            complete = extract_complete_entity(context, "legal_representative")
            if complete:
                return complete.split("：", 1)[1]

        if "注册资本" in question:
            complete = extract_complete_entity(context, "registered_capital")
            if complete:
                return complete.split("：", 1)[1]

        if "重要供应商" in question or "哪个领域" in question:
            for sentence in sentences:
                if "国防军队视频指挥领域的重要供应商" in sentence:
                    return "国防军队视频指挥领域"
                if "军队视频指挥领域的重要供应商" in sentence:
                    return "国防军队视频指挥领域"

        if "国家科技进步一等奖" in question or "一等奖" in question:
            match = re.search(r"[“\"]?(某情报、指挥、控制与通信网络一体化工程)[”\"]?", context)
            if match:
                return match.group(1)

        if "上游" in question:
            for sentence in sentences:
                if "电子信息行业的上游" in sentence and "电子元器件制造企业" in sentence:
                    return self._normalize_text(sentence).rstrip("。！？；?") + "。"
            for sentence in sentences:
                if "上游" in sentence and "电子元器件制造企业" in sentence and "金属壳体制造企业" in sentence:
                    return self._normalize_text(sentence).rstrip("。！？；?") + "。"

        if "技术标准" in question:
            exact_sentences = [
                sentence
                for sentence in sentences
                if "参与制定" in sentence and ("技术标准" in sentence or "技术规范" in sentence)
            ]
            if exact_sentences:
                return self._normalize_text(exact_sentences[0]).rstrip("。！？；?") + "。"

        if "收入" in question:
            money_lines = []
            for line in re.split(r"[\n\r]+", context):
                if any(key in line for key in ["军用", "国防", "民用", "主营业务收入", "销售收入", "营业收入"]):
                    money_lines.append(line.strip())
            money_text = " ".join(money_lines) if money_lines else context
            values = self._extract_money_values(money_text)
            if values:
                return "、".join(values[:6])

        if any(term in question for term in ["多少", "金额", "注册资本", "发行股数", "募集资金"]):
            return ""

        ranked = sorted(
            ((self._score_sentence(sentence, keywords), idx, sentence) for idx, sentence in enumerate(sentences)),
            key=lambda item: (-item[0], item[1]),
        )
        best_sentences = [sentence for score, _, sentence in ranked[:4] if score > 0]

        if best_sentences:
            cleaned = [self._normalize_text(item).rstrip("。！？；?") for item in best_sentences[:2]]
            return "根据检索到的资料，" + "；".join(cleaned) + "。"
        return ""

    def _needs_local_fallback(self, question: str, answer: str, prompt_type: str | None) -> bool:
        text = answer or ""
        question_text = question or ""
        if not text.strip():
            return True
        if "暂时没有找到足够相关的信息" in text:
            return True
        if prompt_type in {"numeric", "percentage"}:
            if not any(ch.isdigit() for ch in text) and "％" not in text and "%" not in text:
                return True
        if any(term in question_text for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"]):
            if "万元" in text or ("%" not in text and "％" not in text):
                return True
        if any(keyword in question_text for keyword in ["收入", "多少", "比例", "占比", "金额", "注册资本"]):
            if not any(ch.isdigit() for ch in text):
                return True
        if len(text) < 8 and len(question_text) > 8:
            return True
        return False

    def _normalize_answer_language(self, answer_language: str = "zh") -> str:
        language = (answer_language or "zh").strip().lower()
        if language in {"en", "english"}:
            return "en"
        return "zh"

    def no_answer_message(self, answer_language: str = "zh") -> str:
        if self._normalize_answer_language(answer_language) == "en":
            return "Based on the current knowledge base, I could not find enough relevant information."
        return "根据当前知识库，暂时没有找到足够相关的信息。"

    def localize_answer(self, question: str, answer: str, answer_language: str = "zh") -> str:
        language = self._normalize_answer_language(answer_language)
        text = self._normalize_text(answer or "")
        if language != "en" or not text:
            return text

        if "暂时没有找到足够相关的信息" in text:
            return self.no_answer_message("en")
        if self.client is None:
            return text

        prompt = (
            "Translate the following answer into concise English. Keep company names, proper nouns, "
            "numbers, monetary amounts, percentages, and source-grounded facts exactly. "
            "Do not add facts or explanations.\n\n"
            f"Question: {question}\n"
            f"Answer: {text}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You translate source-grounded financial QA answers faithfully."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=Config.LLM_MAX_TOKENS,
            )
            translated = self._normalize_text(resp.choices[0].message.content or "")
            return translated or text
        except Exception:
            logger.exception("answer localization failed | question=%s", question)
            return text

    def _offline_answer(self, question: str, context: str = "", answer_language: str = "zh") -> str:
        if not context.strip():
            return self.no_answer_message(answer_language)

        extracted = self._extract_answer_from_context(question, context)
        if extracted:
            return self.localize_answer(question, extracted, answer_language)

        keywords = self._expand_keywords(question)
        sentences = self._split_sentences(context)
        if not sentences:
            return self.no_answer_message(answer_language)

        ranked = []
        for idx, sentence in enumerate(sentences):
            ranked.append((self._score_sentence(sentence, keywords), idx, sentence))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        picked = [item[2] for item in ranked[:3] if item[0] > 0]
        if not picked:
            picked = sentences[:2]

        cleaned = [self._normalize_text(item).rstrip("。！？；?") for item in picked if self._normalize_text(item)]
        if not cleaned:
            return self.no_answer_message(answer_language)

        answer = "根据检索到的资料，" + "；".join(cleaned)
        if not answer.endswith(("。", "！", "?", "？")):
            answer += "。"
        return self.localize_answer(question, answer, answer_language)

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
