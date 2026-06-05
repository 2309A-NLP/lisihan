# -*- coding: utf-8 -*-
# 人工智能 NLP-RAG-基于 PDF 文档的问答系统
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""Document metadata cache and metadata-based answer helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.processing.query_rewriter import _is_liyuan_document
from src.utils.text_utils import _extract_company_entities, _normalize_question_text


class MetadataCache:
    def __init__(self, pdf_parser, logger):
        self.pdf_parser = pdf_parser
        self.logger = logger
        self.document_metadata: Dict[str, str] = {}
        self.document_metadata_by_company: Dict[str, Dict[str, str]] = {}

    def _select_document_metadata(self, question: str) -> Dict[str, str]:
        q = _normalize_question_text(question)
        if not self.document_metadata and not self.document_metadata_by_company:
            self._refresh_document_metadata()
        for company_name, metadata in self.document_metadata_by_company.items():
            if company_name and company_name in q:
                return metadata
        if "力源信息" in q or "武汉力源" in q or "力源信息技术" in q:
            for company_name, metadata in self.document_metadata_by_company.items():
                if "力源信息" in company_name:
                    return metadata
        if "兴图新科" in q or "武汉兴图" in q:
            for company_name, metadata in self.document_metadata_by_company.items():
                if "兴图新科" in company_name:
                    return metadata
        return self.document_metadata

    def _metadata_matches_question(self, question: str, metadata: Dict[str, str]) -> bool:
        companies = _extract_company_entities(_normalize_question_text(question))
        if not companies:
            return True
        company_name = metadata.get("company_name", "")
        return any(company and (company in company_name or company_name in company) for company in companies)

    def _metadata_answer(self, question: str) -> str:
        q = _normalize_question_text(question)
        metadata = self._select_document_metadata(question)
        if not self._metadata_matches_question(question, metadata):
            return ""
        if "法定代表人" in q:
            return metadata.get("legal_representative", "")
        if "注册资本" in q:
            return metadata.get("registered_capital", "")
        if "成立日期" in q or "成立时间" in q:
            return metadata.get("establishment_date", "")
        if "注册地址" in q or "注册地" in q:
            return metadata.get("registered_address", "")
        if "公司名称" in q:
            return metadata.get("company_name", "")
        if _is_liyuan_document(q):
            if "不存在控制关系" in q:
                return self._format_liyuan_non_control_related(metadata)
            if ("本次发行股数" in q or "发行股数" in q) and ("发行后总股本" in q or "比例" in q or "占" in q):
                total_shares = metadata.get("total_shares", "")
                ratio = metadata.get("post_issuance_ratio", "")
                if total_shares and ratio:
                    return f"{total_shares}，占发行后总股本的比例为{ratio}"
            if "本次发行股数" in q or "发行股数" in q:
                return metadata.get("total_shares", "")
            if "发行后总股本" in q or "占发行后总股本" in q:
                return metadata.get("post_issuance_ratio", "")
            if "存在控制关系" in q or "控制关系" in q:
                return self._format_liyuan_control_related(metadata)
        return ""

    def _format_liyuan_control_related(self, metadata: Dict[str, str]) -> str:
        party = metadata.get("control_related_party", "")
        share = metadata.get("control_related_party_share", "")
        relation = metadata.get("control_related_party_relationship", "")
        if party and share and relation:
            return f"{party}，持股比例{share}，{relation}"
        return ""

    def _format_liyuan_non_control_related(self, metadata: Dict[str, str]) -> str:
        parties = metadata.get("non_control_related_parties", "")
        return parties

    def _iter_metadata_files(self):
        for root in [Path(self.pdf_parser.output_dir), Path("parsed_output")]:
            if root.exists():
                yield from sorted(root.glob("*_metadata.json"))

    def _load_document_metadata_cache(self) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        for metadata_file in self._iter_metadata_files():
            try:
                metadata.update(json.loads(metadata_file.read_text(encoding="utf-8")))
            except Exception as exc:
                self.logger.warning("document metadata load failed | file=%s | error=%s", metadata_file, exc)
        return metadata

    def _load_document_metadata_cache_by_company(self) -> Dict[str, Dict[str, str]]:
        metadata_by_company: Dict[str, Dict[str, str]] = {}
        for metadata_file in self._iter_metadata_files():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                company_name = metadata.get("company_name", "")
                if company_name:
                    metadata_by_company[company_name] = metadata
            except Exception as exc:
                self.logger.warning("document metadata load failed | file=%s | error=%s", metadata_file, exc)
        return metadata_by_company

    def _refresh_document_metadata(self) -> None:
        parser_metadata_by_company = getattr(self.pdf_parser, "document_metadata_by_company", {}) or {}
        self.document_metadata_by_company.update(parser_metadata_by_company)
        parser_metadata = getattr(self.pdf_parser, "document_metadata", {}) or {}
        if parser_metadata and not self.document_metadata:
            self.document_metadata.update(parser_metadata)
        if not self.document_metadata_by_company:
            self.document_metadata_by_company.update(self._load_document_metadata_cache_by_company())
        if not self.document_metadata:
            cached_by_company = next(iter(self.document_metadata_by_company.values()), {})
            self.document_metadata.update(cached_by_company or self._load_document_metadata_cache())

    def _metadata_source_chunk(self, question: str) -> Dict:
        selected_metadata = self._select_document_metadata(question)
        return {"content": "document_metadata", "metadata": selected_metadata, "relevance_score": 1.0}
