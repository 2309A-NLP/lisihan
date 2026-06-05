# -*- coding: utf-8 -*-
"""Extract fixed document metadata from prospectus PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import fitz


XINGTU_COMPANY_NAME = "武汉兴图新科电子股份有限公司"
LIYUAN_COMPANY_NAME = "武汉力源信息技术股份有限公司"


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    return text.replace("：", ":")


def _clean_company_name(value: str) -> str:
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9（）()·]+?(?:股份有限公司|有限责任公司|有限公司))", value or "")
    return match.group(1) if match else (value or "").strip("0123456789、.． ")


def _format_money(value: str) -> str:
    value = value.replace(" ", "").replace("人民币", "").replace("万元", "").replace(".00", "")
    return f"{value}万元"


def _is_name_cell(value: str) -> bool:
    value = re.sub(r"\s+", "", value or "")
    if not re.fullmatch(r"[\u4e00-\u9fa5A-Za-z（）()·]{2,24}", value):
        return False
    if "招股" in value or "意向书" in value:
        return False
    skip_terms = {
        "关联方名称",
        "企业名称",
        "与本公司关系",
        "持股比例",
        "公司控股股东",
    }
    if value in skip_terms:
        return False
    return not any(term in value for term in ["持有", "控制", "股东", "公司关系", "实际控制人"])


def _lines_between(text: str, start_marker: str, end_marker: str | None = None) -> list[str]:
    start = (text or "").find(start_marker)
    if start < 0:
        return []
    body = text[start + len(start_marker) :]
    if end_marker:
        end = body.find(end_marker)
        if end >= 0:
            body = body[:end]
    return [line.strip() for line in body.splitlines() if line.strip()]


def _extract_control_related_party(raw_text: str) -> Dict[str, str]:
    lines = _lines_between(raw_text, "1、存在控制关系的关联方", "2、不存在控制关系的关联方")
    metadata: Dict[str, str] = {}
    for index, line in enumerate(lines):
        if not _is_name_cell(line):
            continue
        share = ""
        relation = ""
        for next_line in lines[index + 1 : index + 5]:
            if not share and re.fullmatch(r"\d+(?:\.\d+)?%", next_line):
                share = next_line
                continue
            if share and next_line and not re.fullmatch(r"\d+(?:\.\d+)?%", next_line):
                relation = next_line
                break
        metadata["control_related_party"] = line
        if share:
            metadata["control_related_party_share"] = share
        if relation:
            metadata["control_related_party_relationship"] = relation
        break
    return metadata


def _extract_non_control_related_parties(raw_text: str) -> str:
    lines = _lines_between(raw_text, "2、不存在控制关系的关联方", "3、报告期内曾为关联方")
    names = []
    for line in lines:
        candidate = re.sub(r"\s+", "", line)
        if _is_name_cell(candidate):
            names.append(candidate)
    seen = set()
    return "、".join(name for name in names if not (name in seen or seen.add(name)))


def _extract_issuer_basic_section(raw_text: str, company_name: str) -> str:
    for marker in ["一、发行人的基本情况", "一、发行人简介", "（一）发行人基本情况", "(一)公司概况"]:
        start = 0
        while True:
            index = raw_text.find(marker, start)
            if index < 0:
                break
            section = raw_text[index : index + 2200]
            if "法定代表人" in section and "注册资本" in section:
                return section
            start = index + len(marker)
    sections = []
    for marker in ["一、发行人的基本情况", "一、发行人简介", "（一）发行人基本情况", "(一)公司概况"]:
        start = raw_text.find(marker)
        if start >= 0:
            sections.append(raw_text[start : start + 2200])
    company_sections = [section for section in sections if company_name and company_name in section]
    if company_sections:
        return company_sections[0]
    return sections[0] if sections else raw_text[:5000]


def _extract_labeled_value(raw_text: str, label: str, value_pattern: str, stop_labels: str = "") -> str:
    compact = _normalize(raw_text)
    if stop_labels:
        pattern = rf"{label}:?({value_pattern})(?:{stop_labels})"
    else:
        pattern = rf"{label}:?({value_pattern})"
    match = re.search(pattern, compact)
    return match.group(1) if match else ""


def _extract_capital(raw_text: str) -> str:
    compact = _normalize(raw_text)
    patterns = [
        r"注册资本:?(?:人民币)?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)万元",
        r"注册资本((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)万元",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            return _format_money(match.group(1))
    return ""


def extract_document_metadata(pdf_path: str | Path, max_pages: int = 80) -> Dict[str, str]:
    """Extract stable issuer metadata from the first pages of the prospectus."""
    pdf_file = Path(pdf_path)
    with fitz.open(pdf_file) as doc:
        first_page_texts = [doc[i].get_text("text") for i in range(min(max_pages, len(doc)))]
        all_page_texts = [page.get_text("text") for page in doc]

    raw_text = "\n".join(all_page_texts)
    text = _normalize("\n".join(first_page_texts))
    full_text = _normalize(raw_text)
    metadata: Dict[str, str] = {}

    company_match = re.search(r"公司名称:([^英\n]+?)英文名称", text)
    if company_match:
        metadata["company_name"] = _clean_company_name(company_match.group(1))
    elif "武汉兴图新科电子股份有限公司" in text:
        metadata["company_name"] = XINGTU_COMPANY_NAME
    elif "武汉力源信息技术股份有限公司" in text:
        metadata["company_name"] = LIYUAN_COMPANY_NAME

    issuer_section = _extract_issuer_basic_section(raw_text, metadata.get("company_name", ""))
    legal_value = _extract_labeled_value(
        issuer_section,
        "法定代表人",
        r"[\u4e00-\u9fa5A-Za-z·]{2,20}?",
        "注册资本|实收资本|注册地址|公司住所|住所|5、|6、|\n",
    )
    if legal_value:
        metadata["legal_representative"] = legal_value

    capital = _extract_capital(issuer_section)
    if capital:
        metadata["registered_capital"] = capital

    date_match = re.search(r"兴图新科有限成立日期:(\d{4}年\d{1,2}月\d{1,2}日)", text)
    if date_match:
        metadata["establishment_date"] = date_match.group(1)

    address_match = re.search(r"注册地址:(.+?)(?:兴图新科有限成立日期|邮政编码|电话)", text)
    if address_match:
        metadata["registered_address"] = address_match.group(1)

    if metadata.get("company_name") == LIYUAN_COMPANY_NAME or LIYUAN_COMPANY_NAME in full_text:
        total_shares_match = re.search(r"发行股数.*?([\d,]+万股).*?占发行后总股本的比例为([\d.]+%)", full_text)
        if total_shares_match:
            metadata["total_shares"] = total_shares_match.group(1)
            metadata["post_issuance_ratio"] = total_shares_match.group(2)

        metadata.update(_extract_control_related_party(raw_text))
        non_control_parties = _extract_non_control_related_parties(raw_text)
        if non_control_parties:
            metadata["non_control_related_parties"] = non_control_parties
    return metadata
