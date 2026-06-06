# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import Dict, List


class TextToolsMixin:
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
