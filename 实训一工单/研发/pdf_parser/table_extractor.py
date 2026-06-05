# -*- coding: utf-8 -*-
"""pdfplumber based table extraction."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, List, Sequence

import pdfplumber

from .filters import bbox_to_list, normalize_text


DEFAULT_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
    "text_tolerance": 3,
}


def _clean_table_rows(rows: Sequence[Sequence[object]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for row in rows or []:
        cleaned_row = [normalize_text("" if cell is None else str(cell)) for cell in (row or [])]
        if any(cell for cell in cleaned_row):
            cleaned.append(cleaned_row)
    return cleaned


def _table_to_markdown(rows: Sequence[Sequence[object]]) -> str:
    cleaned = _clean_table_rows(rows)
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    normalized = [row + [""] * (width - len(row)) for row in cleaned]
    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_to_csv(rows: Sequence[Sequence[object]]) -> str:
    cleaned = _clean_table_rows(rows)
    if not cleaned:
        return ""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(cleaned)
    return buffer.getvalue().strip()


def _convert_table_content(rows: Sequence[Sequence[object]], table_to_text: bool, table_format: str) -> object:
    if not table_to_text:
        return _clean_table_rows(rows)

    if table_format == "csv":
        return _table_to_csv(rows)
    return _table_to_markdown(rows)


def extract_table_blocks(
    pdf_path: str | Path,
    table_to_text: bool = False,
    table_format: str = "markdown",
) -> List[Dict]:
    pdf_file = Path(pdf_path)
    results: List[Dict] = []

    with pdfplumber.open(pdf_file) as doc:
        for page_number, page in enumerate(doc.pages, start=1):
            try:
                raw_tables = page.extract_tables(table_settings=DEFAULT_TABLE_SETTINGS) or []
            except Exception:
                raw_tables = []

            table_objects = []
            try:
                table_objects = page.find_tables(table_settings=DEFAULT_TABLE_SETTINGS) or []
            except Exception:
                table_objects = []

            for table_index, rows in enumerate(raw_tables, start=1):
                cleaned_rows = _clean_table_rows(rows)
                if not cleaned_rows:
                    continue

                if table_index - 1 < len(table_objects):
                    bbox = bbox_to_list(table_objects[table_index - 1].bbox)
                else:
                    bbox = [0.0, 0.0, float(page.width), float(page.height)]

                content = _convert_table_content(cleaned_rows, table_to_text=table_to_text, table_format=table_format)
                char_count = len(content) if isinstance(content, str) else sum(
                    len(cell) for row in cleaned_rows for cell in row
                )
                results.append(
                    {
                        "page": page_number,
                        "type": "table",
                        "content": content,
                        "bbox": bbox,
                        "metadata": {
                            "source_file": pdf_file.name,
                            "source_path": str(pdf_file),
                            "table_index": table_index,
                            "char_count": char_count,
                            "has_table": True,
                            "row_count": len(cleaned_rows),
                            "column_count": max((len(row) for row in cleaned_rows), default=0),
                            "table_format": table_format if table_to_text else "raw",
                        },
                    }
                )

    return results

