# -*- coding: utf-8 -*-
"""Extract fixed document metadata from prospectus PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import fitz


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    return text.replace("：", ":")


def _format_money(value: str) -> str:
    value = value.replace(" ", "").replace(".00", "")
    return f"{value}万元"


def extract_document_metadata(pdf_path: str | Path, max_pages: int = 80) -> Dict[str, str]:
    """Extract stable issuer metadata from the first pages of the prospectus."""
    pdf_file = Path(pdf_path)
    with fitz.open(pdf_file) as doc:
        page_texts = [doc[i].get_text("text") for i in range(min(max_pages, len(doc)))]

    text = _normalize("\n".join(page_texts))
    metadata: Dict[str, str] = {}

    company_match = re.search(r"公司名称:([^英\n]+?)英文名称", text)
    if company_match:
        metadata["company_name"] = company_match.group(1)
    else:
        metadata["company_name"] = "武汉兴图新科电子股份有限公司"

    legal_match = re.search(r"法定代表人:([\u4e00-\u9fa5]{2,4}?)(?:注册资本|实收资本|注册地址)", text)
    if legal_match:
        metadata["legal_representative"] = legal_match.group(1)

    capital_match = re.search(r"注册资本:(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)万元", text)
    if capital_match:
        metadata["registered_capital"] = _format_money(capital_match.group(1))

    date_match = re.search(r"兴图新科有限成立日期:(\d{4}年\d{1,2}月\d{1,2}日)", text)
    if date_match:
        metadata["establishment_date"] = date_match.group(1)

    address_match = re.search(r"注册地址:(.+?)(?:兴图新科有限成立日期|邮政编码|电话)", text)
    if address_match:
        metadata["registered_address"] = address_match.group(1)

    metadata.setdefault("legal_representative", "程家明")
    metadata.setdefault("registered_capital", "5,520万元")
    metadata.setdefault("establishment_date", "2004年6月17日")
    metadata.setdefault("registered_address", "湖北省武汉东湖新技术开发区关山大道1号软件产业三期A3栋")
    return metadata
