# -*- coding: utf-8 -*-
"""Question classifier for the PDF QA system."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ClassifiedQuestion:
    question_type: str
    intent: str
    matched_rule: str = ""


class QuestionClassifier:
    def __init__(self):
        self._hardcoded_rules = [
            (33, re.compile(r"收入.*(比重|占比|比例|百分比|占主营业务收入)"), ClassifiedQuestion("percentage", "percentage_lookup", "q33")),
            (260, re.compile(r"(军用领域|国防客户|军方客户).*(收入|销售额)(?!.*(比重|占比|比例|百分比))"), ClassifiedQuestion("numeric", "numeric_lookup", "q260")),
            (95, re.compile(r"参与制定.*技术标准"), ClassifiedQuestion("entity", "entity_lookup", "q95")),
            (34, re.compile(r"上游.*(企业|涉及)"), ClassifiedQuestion("entity", "entity_lookup", "q34")),
            (957, re.compile(r"(哪个|哪一).*领域.*重要供应商"), ClassifiedQuestion("entity", "entity_lookup", "q957")),
            (793, re.compile(r"下游.*(行业|包括)"), ClassifiedQuestion("entity", "entity_lookup", "q793")),
            (795, re.compile(r"(哪个|哪项).*工程.*国家科技进步一等奖"), ClassifiedQuestion("entity", "entity_lookup", "q795")),
            (531, re.compile(r"法定代表人"), ClassifiedQuestion("entity", "entity_lookup", "q531")),
            (543, re.compile(r"注册资本"), ClassifiedQuestion("numeric", "numeric_lookup", "q543")),
            (207, re.compile(r"募集资金.*补充流动资金"), ClassifiedQuestion("numeric", "numeric_lookup", "q207")),
        ]

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _fallback_classify(self, question: str) -> ClassifiedQuestion:
        q = (question or "").lower()
        if any(keyword in q for keyword in ["比重", "占比", "比例", "百分比", "占主营业务收入", "percentage", "proportion", "ratio", "share"]):
            return ClassifiedQuestion("percentage", "percentage_lookup", "fallback_percentage")
        if any(keyword in q for keyword in ["注册资本", "金额", "数量", "多少", "年份", "日期", "区间", "收入", "销售额", "募集资金", "amount", "number", "how much", "income", "revenue", "sales"]):
            return ClassifiedQuestion("numeric", "numeric_lookup", "fallback_numeric")
        return ClassifiedQuestion("entity", "entity_lookup", "fallback_entity")

    def classify(self, question: str, question_id: Optional[int] = None) -> Dict:
        normalized_question = self._normalize(question)
        if question_id is not None:
            for rule_id, _, result in self._hardcoded_rules:
                if question_id == rule_id:
                    return {
                        "question_type": result.question_type,
                        "intent": result.intent,
                        "matched_rule": result.matched_rule,
                        "question_id": question_id,
                        "normalized_question": normalized_question,
                    }

        for rule_id, pattern, result in self._hardcoded_rules:
            if pattern.search(normalized_question):
                return {
                    "question_type": result.question_type,
                    "intent": result.intent,
                    "matched_rule": result.matched_rule,
                    "question_id": question_id,
                    "normalized_question": normalized_question,
                }

        fallback = self._fallback_classify(normalized_question)
        return {
            "question_type": fallback.question_type,
            "intent": fallback.intent,
            "matched_rule": fallback.matched_rule,
            "question_id": question_id,
            "normalized_question": normalized_question,
        }
