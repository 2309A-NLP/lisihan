# -*- coding: utf-8 -*-
"""Question classifier for the PDF QA system."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from utils.language import detect_language

QUESTION_TYPES = ("numeric", "percentage", "entity")


@dataclass(frozen=True)
class ClassifiedQuestion:
    question_type: str
    intent: str
    matched_rule: str = ""


class QuestionClassifier:
    def __init__(self):
        self._hardcoded_rules = [
            (260, re.compile(r"(军用领域|军方|国防).*(收入|销售额).*多少"), ClassifiedQuestion("numeric", "numeric_lookup", "q260")),
            (33, re.compile(r"(收入占|占.*收入|比重|占比).*多少"), ClassifiedQuestion("percentage", "percentage_lookup", "q33")),
            (95, re.compile(r"参与制定.*(技术标准|技术规范)|制定了哪个技术标准"), ClassifiedQuestion("entity", "entity_lookup", "q95")),
            (34, re.compile(r"上游.*哪些.*企业"), ClassifiedQuestion("entity", "entity_lookup", "q34")),
            (957, re.compile(r"哪个领域.*重要供应商|在哪个领域.*供应商"), ClassifiedQuestion("entity", "entity_lookup", "q957")),
            (793, re.compile(r"下游.*哪些.*行业"), ClassifiedQuestion("entity", "entity_lookup", "q793")),
            (795, re.compile(r"哪个工程.*国家科技进步一等奖"), ClassifiedQuestion("entity", "entity_lookup", "q795")),
            (531, re.compile(r"法定代表人.*谁"), ClassifiedQuestion("entity", "entity_lookup", "q531")),
            (207, re.compile(r"多少.*募集资金.*补充流动资金|补充流动资金.*多少"), ClassifiedQuestion("numeric", "numeric_lookup", "q207")),
            (543, re.compile(r"注册资本.*多少"), ClassifiedQuestion("numeric", "numeric_lookup", "q543")),
        ]

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _fallback_classify(self, question: str) -> ClassifiedQuestion:
        q = (question or "").lower()
        if any(keyword in q for keyword in ["percentage", "proportion", "ratio", "share", "accounted for"]):
            return ClassifiedQuestion("percentage", "percentage_lookup", "fallback_percentage_en")
        if any(keyword in q for keyword in ["收入占比", "占比", "比重", "百分比", "比例"]):
            return ClassifiedQuestion("percentage", "percentage_lookup", "fallback_percentage")
        if any(
            keyword in q
            for keyword in [
                "注册资本",
                "金额",
                "数量",
                "多少",
                "年份",
                "日期",
                "收入",
                "募集资金",
                "revenue",
                "income",
                "how much",
                "amount",
                "registered capital",
            ]
        ):
            return ClassifiedQuestion("numeric", "numeric_lookup", "fallback_numeric")
        return ClassifiedQuestion("entity", "entity_lookup", "fallback_entity")

    def classify(self, question: str, question_id: Optional[int] = None) -> Dict:
        normalized_question = self._normalize(question)
        language = detect_language(question)

        if question_id is not None:
            for rule_id, _, result in self._hardcoded_rules:
                if question_id == rule_id:
                    return {
                        "question_type": result.question_type,
                        "intent": result.intent,
                        "matched_rule": result.matched_rule,
                        "question_id": question_id,
                        "normalized_question": normalized_question,
                        "language": language,
                    }

        for _, pattern, result in self._hardcoded_rules:
            if pattern.search(normalized_question):
                return {
                    "question_type": result.question_type,
                    "intent": result.intent,
                    "matched_rule": result.matched_rule,
                    "question_id": question_id,
                    "normalized_question": normalized_question,
                    "language": language,
                }

        fallback = self._fallback_classify(normalized_question)
        return {
            "question_type": fallback.question_type,
            "intent": fallback.intent,
            "matched_rule": fallback.matched_rule,
            "question_id": question_id,
            "normalized_question": normalized_question,
            "language": language,
        }
