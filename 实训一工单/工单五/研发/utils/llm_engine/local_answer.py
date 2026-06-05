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
from utils.question_classifier import QuestionClassifier

from .entities import extract_complete_entity

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - 可选依赖
    OpenAI = None


logger = get_logger(__name__)


class LocalAnswerMixin:
    def postprocess_answer(self, question: str, answer: str) -> str:
        """清洗模型输出中的表格残片、无关上下文和多余数值。"""
        question = question or ""
        text = self._normalize_text(answer or "")
        text = re.sub(r"\|+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" ，,；;。")

        if "关联方" in question and "存在控制关系" in question and "不存在控制关系" not in question:
            related_party_answer = self._extract_control_related_party_answer(question, answer)
            if related_party_answer:
                return related_party_answer
            if "持股比例" in text and "%" in text:
                return text

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
            supplier_field = self._extract_supplier_field(text)
            if supplier_field:
                return supplier_field

        if "国家科技进步一等奖" in question or "一等奖" in question:
            award_project = self._extract_award_project(text)
            if award_project:
                return award_project

        if any(term in question for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"]):
            values = self._extract_percentage_values(text)
            if values:
                limit = 3 if "2019" not in question and "1-6" not in question and "1至6" not in question else 4
                return "、".join(values[:limit])

        return text

    def _extract_answer_from_context(self, question: str, context: str) -> str:
        if not context.strip():
            return ""

        keywords = self._expand_keywords(question)
        sentences = self._split_sentences(context)

        focused_numeric = self._extract_focused_numeric_answer(question, context)
        if focused_numeric and ("发行股数" in question or "发行后总股本" in question):
            return focused_numeric

        related_party_answer = self._extract_control_related_party_answer(question, context)
        if related_party_answer:
            return related_party_answer

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
                supplier_field = self._extract_supplier_field(sentence)
                if supplier_field:
                    return supplier_field

        if "国家科技进步一等奖" in question or "一等奖" in question:
            award_project = self._extract_award_project(context)
            if award_project:
                return award_project

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

    def _extract_control_related_party_answer(self, question: str, context: str) -> str:
        q = question or ""
        if "关联方" not in q or "存在控制关系" not in q or "不存在控制关系" in q:
            return ""
        for raw_line in re.split(r"[\r\n]+", context or ""):
            line = raw_line.strip()
            if not line.startswith("|") or "|" not in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell and set(cell) != {"-"}]
            if len(cells) < 3 or cells[0] in {"关联方名称", "企业名称"}:
                continue
            if "%" in cells[1] and "关系" not in cells[2]:
                return f"{cells[0]}，持股比例{cells[1]}，{cells[2]}"
        return ""

    def _extract_supplier_field(self, text: str) -> str:
        match = re.search(r"成为([^。；;，,]{2,40}?领域)的重要供应商", text or "")
        if match:
            return match.group(1)
        match = re.search(r"([^。；;，,]{2,40}?领域)的重要供应商", text or "")
        if match:
            return match.group(1)
        return ""

    def _extract_award_project(self, text: str) -> str:
        patterns = [
            r"[“\"]([^”\"]{4,80}?工程)[”\"]荣获国家科技进步一等奖",
            r"([^。；;，,]{4,80}?工程)荣获国家科技进步一等奖",
            r"荣获国家科技进步一等奖的?([^。；;，,]{4,80}?工程)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text or "")
            if match:
                return match.group(1).strip("“”\" ")
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
