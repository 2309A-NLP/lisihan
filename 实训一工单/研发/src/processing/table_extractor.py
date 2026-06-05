# -*- coding: utf-8 -*-
# 人工智能 NLP-RAG-基于 PDF 文档的问答系统
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
"""Table-derived answer extraction helpers."""

from __future__ import annotations

import re


def _clean_project_name(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    value = value.strip("|-—、，,。；;:：")
    return value


def _extract_liyuan_table_answer(question: str, context: str) -> str:
    q = question or ""
    if "募集资金拟投资" in q or "募集资金用途" in q:
        projects = []
        prose_match = re.search(r"本次募集资金拟投资以下项目[:：]\s*([^。\n]+)", context or "")
        if prose_match:
            for item in re.split(r"[、,，；;]", prose_match.group(1)):
                project = _clean_project_name(item)
                if project and not re.search(r"^\d", project):
                    projects.append(project)

        capture = False
        for raw_line in (context or "").splitlines():
            line = raw_line.strip()
            if "本次募集资金拟投资以下项目" in line or "项目名称" in line:
                capture = True
            if not capture or not line.startswith("|"):
                continue
            cells = [_clean_project_name(cell) for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell and cell != "---"]
            if not cells or any(cell in {"项目名称", "序号", "计划总投资(万元)"} for cell in cells):
                continue
            candidate = ""
            for cell in cells:
                if re.fullmatch(r"\d+(?:\.\d+)?", cell) or re.search(r"万元|^\d+$", cell):
                    continue
                if len(cell) >= 2 and re.search(r"[\u4e00-\u9fff]", cell):
                    candidate = cell
                    break
            if candidate:
                projects.append(candidate)
        if projects:
            seen = set()
            cleaned_projects = []
            for project in projects:
                project = _clean_project_name(project)
                if not project or project in seen:
                    continue
                seen.add(project)
                cleaned_projects.append(project)
            return "、".join(cleaned_projects)
    return ""
