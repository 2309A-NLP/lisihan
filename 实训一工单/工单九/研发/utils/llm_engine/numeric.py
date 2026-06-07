# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import List


class NumericExtractionMixin:
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

    def _is_military_income_amount_question(self, question: str) -> bool:
        q = question or ""
        is_military_income = any(term in q for term in ["军用领域", "国防客户", "军方客户", "军品业务"])
        is_income_amount = "收入" in q or "销售额" in q
        is_percentage = any(term in q for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"])
        return is_military_income and is_income_amount and not is_percentage
