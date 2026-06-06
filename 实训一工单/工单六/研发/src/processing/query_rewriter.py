# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from src.utils.text_utils import _normalize_question_text


def _is_liyuan_document(question: str) -> bool:
    q = _normalize_question_text(question)
    return any(keyword in q for keyword in ["武汉力源", "力源信息", "力源信息技术"])


def rewrite_query(question: str) -> str:
    """把业务口径问题改写成招股书中的标准字段，提升召回准确率。"""
    q = _normalize_question_text(question)
    is_military_income = any(term in q for term in ["军用领域", "国防客户", "军方客户", "军品业务"])
    is_income_amount = "收入" in q or "销售额" in q
    is_percentage = any(term in q for term in ["比重", "占比", "比例", "百分比", "占主营业务收入"])
    if is_military_income and is_income_amount and not is_percentage:
        return "直接和间接向国防客户的销售额合计分别是多少"
    if _is_liyuan_document(q) and ("募集资金拟投资" in q or "募集资金用途" in q):
        return "武汉力源信息技术股份有限公司 本次募集资金拟投资以下项目 项目名称 计划总投资"
    if "募集资金" in q and any(term in q for term in ["多少", "金额", "资金", "投入", "用于"]):
        focus_terms = [term for term in ["补充流动资金", "补充营运资金", "拟投入募集资金", "拟使用本次发行募集资金", "项目名称"] if term in q]
        return f"{q} 募集资金 金额 {' '.join(focus_terms)}"
    return q


def select_pdf(question: str) -> str:
    q = _normalize_question_text(question)
    if any(keyword in q for keyword in ["武汉力源", "力源信息", "力源信息技术"]):
        return "招股说明书2.pdf"
    if "武汉兴图" in q or "兴图新科" in q:
        return "招股说明书1.pdf"
    return "招股说明书1.pdf"


def _infer_source_file_from_question(question: str) -> str | None:
    q = _normalize_question_text(question)
    if any(keyword in q for keyword in ["武汉力源", "力源信息", "力源信息技术"]):
        return "招股说明书2.pdf"
    if "武汉兴图" in q or "兴图新科" in q:
        return "招股说明书1.pdf"
    return None
