# -*- coding: utf-8 -*-
"""Convert MinerU outputs to the project's parsed chunk format."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List

from .chunk_strategies import apply_text_chunk_strategy, normalize_chunk_strategy
from .metadata_extractor import extract_document_metadata


def _normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_table_to_markdown(table_html: str) -> str:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html or "", flags=re.IGNORECASE | re.DOTALL)
    parsed_rows: List[List[str]] = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
        clean_cells = [_normalize_text(cell).replace("\n", " ") for cell in cells]
        if clean_cells:
            parsed_rows.append(clean_cells)

    if not parsed_rows:
        return _normalize_text(table_html)

    width = max(len(row) for row in parsed_rows)
    normalized = [row + [""] * (width - len(row)) for row in parsed_rows]
    lines = ["| " + " | ".join(normalized[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _markdown_to_items(markdown_path: Path) -> List[Dict]:
    text = markdown_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text)
    items: List[Dict] = []
    for block in blocks:
        block = block.strip()
        if not block or block.startswith("![]("):
            continue
        items.append({"type": "text", "text": block, "page_idx": 0, "bbox": []})
    return items


def _load_mineru_items(content_list_path: str | None, markdown_path: str | None) -> List[Dict]:
    if content_list_path and Path(content_list_path).exists():
        return json.loads(Path(content_list_path).read_text(encoding="utf-8"))
    if markdown_path and Path(markdown_path).exists():
        return _markdown_to_items(Path(markdown_path))
    return []


def mineru_output_to_parsed_result(
    pdf_path: str | Path,
    *,
    markdown_path: str | None,
    content_list_path: str | None,
    output: str | Path,
    max_text_chars: int = 1500,
    chunk_overlap: int = 0,
    chunk_strategy: str = "paragraph",
) -> Dict:
    pdf_file = Path(pdf_path)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_mineru_items(content_list_path, markdown_path)
    raw_text_blocks: List[Dict] = []
    raw_table_blocks: List[Dict] = []
    table_count = 0
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)

    for item in items:
        item_type = item.get("type", "text")
        if item_type in {"image", "header", "footer"}:
            continue

        if item_type == "table":
            content = _html_table_to_markdown(item.get("table_body", ""))
            chunk_type = "table"
            table_count += 1
        else:
            content = item.get("text") or item.get("content") or ""
            chunk_type = "text"

        page = int(item.get("page_idx", 0)) + 1
        bbox = item.get("bbox") or []
        content = _normalize_text(content)
        if not content:
            continue
        block = {
            "page": page,
            "type": chunk_type,
            "content": content,
            "bbox": bbox,
            "metadata": {
                "source_file": pdf_file.name,
                "source_path": str(pdf_file),
                "parser": "mineru",
                "mineru_type": item_type,
                "char_count": len(content),
                "has_table": chunk_type == "table",
                "chunk_strategy": normalized_strategy,
            },
        }
        if chunk_type == "table":
            raw_table_blocks.append(block)
        else:
            raw_text_blocks.append(block)

    text_chunks = apply_text_chunk_strategy(
        raw_text_blocks,
        strategy=normalized_strategy,
        max_chars=max_text_chars,
        overlap=chunk_overlap,
    )
    chunks: List[Dict] = [*text_chunks, *raw_table_blocks]
    chunks = sorted(
        chunks,
        key=lambda block: (
            int(block.get("page", 0)),
            float((block.get("bbox") or [0, 0, 0, 0])[1]) if block.get("bbox") else 0.0,
            float((block.get("bbox") or [0, 0, 0, 0])[0]) if block.get("bbox") else 0.0,
            0 if block.get("type") == "text" else 1,
        ),
    )
    page_counts: Dict[int, int] = {}
    for chunk in chunks:
        page = int(chunk.get("page", 0))
        page_counts[page] = page_counts.get(page, 0) + 1
        chunk["chunk_id"] = f"p{page}_{page_counts[page]:03d}"
        chunk.setdefault("metadata", {})
        chunk["metadata"]["chunk_strategy"] = normalized_strategy
        chunk["metadata"]["char_count"] = len(str(chunk.get("content", "")))

    document_metadata = extract_document_metadata(pdf_file)
    result = {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "page_count": max((chunk["page"] for chunk in chunks), default=0),
        "chunk_count": len(chunks),
        "table_count": table_count,
        "table_to_text": True,
        "table_format": "markdown",
        "chunk_strategy": normalized_strategy,
        "chunk_size": max_text_chars,
        "chunk_overlap": chunk_overlap,
        "parser": "mineru",
        "mineru_markdown_path": markdown_path,
        "mineru_content_list_path": content_list_path,
        "document_metadata": document_metadata,
        "chunks": chunks,
    }

    output_path = output_dir / f"{pdf_file.stem}_chunks.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path = output_dir / f"{pdf_file.stem}_metadata.json"
    metadata_path.write_text(json.dumps(document_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "output_path": str(output_path)}
