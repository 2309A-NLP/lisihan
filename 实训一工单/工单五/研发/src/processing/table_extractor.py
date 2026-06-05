# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

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
            if projects:
                seen = set()
                return "、".join(project for project in projects if not (project in seen or seen.add(project)))

        capture = False
        for raw_line in (context or "").splitlines():
            line = raw_line.strip()
            if "本次募集资金拟投资以下项目" in line or "项目名称" in line:
                capture = True
            elif line.startswith("---") or "关联方" in line or "企业名称" in line or "与本公司关系" in line:
                capture = False
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
