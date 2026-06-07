# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件用于工单五的指代消解：识别多轮对话中的公司实体和指代词，并将当前
问题改写为适合 RAG 检索的完整问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.utils.text_utils import _normalize_question_text


XINGTU = "武汉兴图新科电子股份有限公司"
LIYUAN = "武汉力源信息技术股份有限公司"


COMPANY_ALIASES: Dict[str, Sequence[str]] = {
    XINGTU: (
        "武汉兴图新科电子股份有限公司",
        "武汉兴图新科",
        "兴图新科电子",
        "兴图新科",
        "武汉兴图",
    ),
    LIYUAN: (
        "武汉力源信息技术股份有限公司",
        "武汉力源信息技术",
        "力源信息技术",
        "力源信息",
        "武汉力源",
        "力源",
    ),
}


COMPANY_PRONOUNS = ("这个公司", "该公司", "那家公司")
PERSON_OR_OBJECT_PRONOUN_PATTERN = re.compile(r"(?<!其)[他她它](?!们)")
SWITCH_COMPANY_PATTERN = re.compile(r"^\s*(?:那|那么|那么，|那，)?(?P<target>.+?)(?:呢|怎么样|如何)?[？?]?\s*$")


@dataclass
class CoreferenceResult:
    original_question: str
    resolved_question: str
    previous_company: str = ""
    current_company: str = ""
    mentioned_companies: List[str] = None
    is_resolved: bool = False
    resolved_mentions: Dict[str, str] = None
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "original_question": self.original_question,
            "resolved_question": self.resolved_question,
            "previous_company": self.previous_company,
            "current_company": self.current_company,
            "mentioned_companies": self.mentioned_companies or [],
            "is_resolved": self.is_resolved,
            "resolved_mentions": self.resolved_mentions or {},
            "reason": self.reason,
        }


class CoreferenceResolver:
    """基于规则的公司指代消解器。"""

    def extract_companies(self, text: str) -> List[str]:
        q = _normalize_question_text(text or "")
        hits = []
        for canonical, aliases in COMPANY_ALIASES.items():
            positions = [q.find(alias) for alias in aliases if alias and alias in q]
            if positions:
                hits.append((min(positions), canonical))
        hits.sort(key=lambda item: item[0])
        companies: List[str] = []
        for _, company in hits:
            if company not in companies:
                companies.append(company)
        return companies

    def latest_company(self, history: Sequence[Dict]) -> str:
        for item in reversed(history or []):
            current_company = item.get("current_company") or ""
            if current_company:
                return current_company

            mentioned = item.get("mentioned_companies") or []
            if mentioned:
                return mentioned[-1]

            for key in ("resolved_question", "question"):
                companies = self.extract_companies(item.get(key, ""))
                if companies:
                    return companies[-1]
        return ""

    def resolve(self, question: str, history: Sequence[Dict] = None) -> CoreferenceResult:
        original_question = _normalize_question_text(question)
        history = history or []
        previous_company = self.latest_company(history)
        mentioned_companies = self.extract_companies(original_question)
        resolved_question = original_question
        resolved_mentions: Dict[str, str] = {}
        reason = ""

        switch_question = self._resolve_switch_question(original_question, mentioned_companies, history)
        if switch_question and switch_question != original_question:
            resolved_question = switch_question
            reason = "switch_company_follow_up"
            resolved_mentions["那...呢"] = mentioned_companies[-1]
            mentioned_companies = self.extract_companies(resolved_question)
        else:
            target_company = mentioned_companies[-1] if mentioned_companies else previous_company
            if target_company:
                for pronoun in COMPANY_PRONOUNS:
                    if pronoun in resolved_question:
                        resolved_question = resolved_question.replace(pronoun, target_company)
                        resolved_mentions[pronoun] = target_company

                if PERSON_OR_OBJECT_PRONOUN_PATTERN.search(resolved_question):
                    resolved_question = PERSON_OR_OBJECT_PRONOUN_PATTERN.sub(target_company, resolved_question)
                    for pronoun in ("他", "她", "它"):
                        if pronoun in original_question:
                            resolved_mentions[pronoun] = target_company

                if resolved_mentions:
                    reason = "pronoun_replaced"

        final_companies = self.extract_companies(resolved_question) or mentioned_companies
        current_company = final_companies[-1] if final_companies else previous_company
        return CoreferenceResult(
            original_question=original_question,
            resolved_question=resolved_question,
            previous_company=previous_company,
            current_company=current_company,
            mentioned_companies=final_companies,
            is_resolved=resolved_question != original_question,
            resolved_mentions=resolved_mentions,
            reason=reason,
        )

    def _resolve_switch_question(self, question: str, mentioned_companies: List[str], history: Sequence[Dict]) -> str:
        if not mentioned_companies:
            return ""
        if "呢" not in question and not question.startswith(("那", "那么")):
            return ""

        target_company = mentioned_companies[-1]
        compact_question = re.sub(r"[\s，,。！？?]", "", question)
        compact_target = re.sub(r"[\s，,。！？?]", "", target_company)
        alias_compacts = [
            re.sub(r"[\s，,。！？?]", "", alias)
            for alias in COMPANY_ALIASES[target_company]
        ]
        is_company_only_follow_up = compact_question in {
            f"那{alias}呢" for alias in alias_compacts
        } | {
            f"{alias}呢" for alias in alias_compacts
        } | {
            f"那么{alias}呢" for alias in alias_compacts
        } | {
            compact_target,
            f"{compact_target}呢",
            f"那{compact_target}呢",
        }
        if not is_company_only_follow_up:
            return ""

        previous_question = self._latest_user_question(history)
        if not previous_question:
            return question

        rewritten = previous_question
        previous_companies = self.extract_companies(previous_question)
        if previous_companies:
            rewritten = self._replace_company_aliases(rewritten, previous_companies[-1], target_company)
        else:
            for pronoun in COMPANY_PRONOUNS:
                rewritten = rewritten.replace(pronoun, target_company)
            rewritten = PERSON_OR_OBJECT_PRONOUN_PATTERN.sub(target_company, rewritten)
            if target_company not in rewritten:
                rewritten = f"{target_company}{rewritten}"
        return rewritten

    def _latest_user_question(self, history: Sequence[Dict]) -> str:
        for item in reversed(history or []):
            question = item.get("resolved_question") or item.get("question") or ""
            if question:
                return _normalize_question_text(question)
        return ""

    def _replace_company_aliases(self, question: str, source_company: str, target_company: str) -> str:
        aliases = sorted(COMPANY_ALIASES.get(source_company, (source_company,)), key=len, reverse=True)
        rewritten = question
        replaced = False
        for alias in aliases:
            if alias in rewritten:
                rewritten = rewritten.replace(alias, target_company)
                replaced = True
        if replaced:
            return rewritten
        return f"{target_company}{rewritten}"
