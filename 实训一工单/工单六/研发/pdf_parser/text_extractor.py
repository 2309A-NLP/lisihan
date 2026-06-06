# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz

from .filters import bbox_to_list, normalize_text, should_drop_text_block


def _block_to_text(block: Dict) -> str:
    lines: List[str] = []
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        line_text = "".join(span.get("text", "") for span in spans)
        line_text = normalize_text(line_text)
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip()


def extract_text_blocks(pdf_path: str | Path) -> List[Dict]:
    pdf_file = Path(pdf_path)
    results: List[Dict] = []

    with fitz.open(pdf_file) as doc:
        for page_number, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            page_rect = page.rect
            for block_index, block in enumerate(page_dict.get("blocks", []), start=1):
                if block.get("type") != 0:
                    continue

                text = _block_to_text(block)
                bbox = bbox_to_list(block.get("bbox"))
                if should_drop_text_block(text, bbox=bbox, page_width=page_rect.width, page_height=page_rect.height):
                    continue

                text = normalize_text(text)
                if len(text) < 5:
                    continue

                results.append(
                    {
                        "page": page_number,
                        "type": "text",
                        "content": text,
                        "bbox": bbox,
                        "metadata": {
                            "source_file": pdf_file.name,
                            "source_path": str(pdf_file),
                            "block_index": block_index,
                            "char_count": len(text),
                            "has_table": False,
                            "page_width": page_rect.width,
                            "page_height": page_rect.height,
                        },
                    }
                )

    return results

