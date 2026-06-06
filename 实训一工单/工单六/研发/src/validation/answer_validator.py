# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import re
from typing import List

from src.constants import NO_ANSWER_EN, NO_ANSWER_ZH, QUALITY_CONFIDENCE_THRESHOLD
from src.utils.text_utils import _chunk_to_text, _extract_company_entities, _normalize_question_text
from src.validation.negative_handler import _is_negative_question


def _question_type_hint(question: str) -> str:
    q = _normalize_question_text(question)
    if any(term in q for term in ["占比", "比重", "比例", "百分比", "占主营业务收入"]):
        return "percentage"
    if any(term in q for term in ["法定代表人", "谁"]):
        return "person"
    if any(term in q for term in ["关联方", "哪些", "有哪些", "项目", "企业", "行业", "标准"]):
        return "list"
    if any(term in q for term in ["金额", "收入", "注册资本", "股数", "多少", "数量"]):
        return "numeric"
    if any(term in q for term in ["是否", "有没有", "存在", "不存在"]):
        return "judgement"
    return "entity"


def _has_control_relation_conflict(question: str, answer: str) -> bool:
    q = _normalize_question_text(question)
    a = answer or ""
    asks_non_control = any(marker in q for marker in ["不存在控制关系", "无控制关系", "不具有控制关系", "没有控制关系"])
    if not asks_non_control:
        return False
    has_non_control_phrase = any(marker in a for marker in ["不存在控制关系", "无控制关系", "不具有控制关系", "没有控制关系"])
    has_positive_control_phrase = any(marker in a for marker in ["存在控制关系", "控制关系的关联方", "控股股东", "实际控制人"])
    return has_positive_control_phrase and not has_non_control_phrase


def _content_words(text: str) -> List[str]:
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9.]+", text or "")
    stop_words = {
        "公司",
        "发行人",
        "本公司",
        "哪个",
        "哪些",
        "什么",
        "多少",
        "是谁",
        "的是",
        "以及",
        "相关",
    }
    return [word for word in words if len(word) >= 2 and word not in stop_words]


def _answer_grounding_score(answer: str, chunk_text: str) -> float:
    compact_answer = re.sub(r"\s+", "", answer or "")
    compact_chunks = re.sub(r"\s+", "", chunk_text or "")
    if not compact_answer or not compact_chunks:
        return 0.0
    if compact_answer[:80] in compact_chunks or compact_answer in compact_chunks:
        return 1.0

    answer_terms = _content_words(compact_answer)
    if not answer_terms:
        return 0.0
    matched = sum(1 for term in answer_terms if term in compact_chunks)
    return matched / max(len(answer_terms), 1)


def validate_answer_quality(question: str, answer: str, retrieved_chunks: list) -> dict:
    """
    校验答案质量。
    返回: {"is_valid": bool, "reason": str, "confidence": float}
    """
    q = _normalize_question_text(question)
    a = re.sub(r"\s+", " ", answer or "").strip()
    chunk_text = " ".join(_chunk_to_text(chunk) for chunk in (retrieved_chunks or []))
    reasons: List[str] = []
    confidence = 1.0

    if not a:
        return {"is_valid": False, "reason": "empty_answer", "confidence": 0.0}
    if NO_ANSWER_ZH in a or NO_ANSWER_EN in a or "没有找到足够相关的信息" in a:
        return {"is_valid": False, "reason": "no_answer", "confidence": 0.0}

    type_hint = _question_type_hint(q)
    if len(a) < 2:
        reasons.append("answer_too_short")
        confidence -= 0.45
    elif len(a) < 4 and type_hint not in {"person", "numeric", "percentage"}:
        reasons.append("answer_short_for_question_type")
        confidence -= 0.25

    grounding_score = _answer_grounding_score(a, chunk_text)
    answer_is_grounded = grounding_score >= 0.6

    for company in _extract_company_entities(q):
        if answer_is_grounded or (type_hint in {"numeric", "percentage", "person"} and (re.search(r"\d|%", a) or len(a) <= 12)):
            continue
        if company not in a and company not in chunk_text:
            reasons.append(f"missing_company_entity:{company}")
            confidence -= 0.25

    if type_hint == "percentage" and not re.search(r"\d+(?:\.\d+)?\s*%|百分之", a):
        reasons.append("percentage_question_without_percentage")
        confidence -= 0.55
    elif type_hint == "numeric" and not re.search(r"\d|一|二|三|四|五|六|七|八|九|十|百|千|万|亿", a):
        reasons.append("numeric_question_without_number")
        confidence -= 0.45
    elif type_hint == "person":
        has_person_like_answer = bool(re.search(r"[\u4e00-\u9fff]{2,4}", a))
        has_wrong_numeric_focus = bool(re.search(r"\d+(?:\.\d+)?\s*(?:万元|元|%|万股)", a))
        if not has_person_like_answer or has_wrong_numeric_focus:
            reasons.append("person_question_type_mismatch")
            confidence -= 0.45
    elif type_hint == "list":
        has_list_signal = any(sign in a for sign in ["、", "，", ",", "；", ";"]) or len(re.findall(r"[\u4e00-\u9fff]{2,}", a)) >= 2
        is_amount_only = bool(re.fullmatch(r"[\d,.%\s万亿元、，；;]+", a))
        short_grounded_entity = answer_is_grounded and len(a) <= 80 and bool(_content_words(a))
        if is_amount_only or (not has_list_signal and not short_grounded_entity):
            reasons.append("list_question_type_mismatch")
            confidence -= 0.35

    if answer_is_grounded and confidence < QUALITY_CONFIDENCE_THRESHOLD:
        reasons.append("grounded_answer_relaxed")
        confidence = max(confidence, QUALITY_CONFIDENCE_THRESHOLD)

    if "关联方" in q:
        related_signal = any(term in a + chunk_text for term in ["关联方", "关联关系", "控制关系", "关联企业", "关联自然人"])
        amount_signal = any(term in a for term in ["收入", "销售额", "主营业务收入", "万元"]) and not related_signal
        if amount_signal:
            reasons.append("related_party_question_answered_with_income")
            confidence -= 0.65
        elif not related_signal and len(a) > 80:
            reasons.append("related_party_context_missing")
            confidence -= 0.25

    if _is_negative_question(q):
        if _has_control_relation_conflict(q, a):
            reasons.append("negative_control_relation_conflict")
            confidence -= 0.75
        if "未披露" in q:
            if any(term in a for term in ["明确披露", "已披露"]):
                reasons.append("undisclosed_question_disclosure_conflict")
                confidence -= 0.45
            concrete_answer = bool(re.search(r"\d", a)) or bool(re.search(r"[\u4e00-\u9fff]{2,}(?:公司|投资|贸易|科技|电子|有限|股份)", a))
            if concrete_answer and "未披露相关信息" not in a:
                reasons.append("undisclosed_question_has_concrete_answer")
                confidence = 0.0

    confidence = max(0.0, min(1.0, round(confidence, 4)))
    return {
        "is_valid": confidence >= QUALITY_CONFIDENCE_THRESHOLD,
        "reason": ";".join(reasons) if reasons else "ok",
        "confidence": confidence,
    }
