# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .parser import parse_pdf_file


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF prospectus parsing and chunking")
    parser.add_argument("pdf_path", nargs="+", help="PDF file path(s) or directory path(s)")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file or directory")
    parser.add_argument("--table_to_text", action="store_true", help="Convert tables to readable text")
    parser.add_argument(
        "--table_format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Table text format",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=1500,
        help="Maximum characters per text chunk",
    )
    parser.add_argument(
        "--no_dedup_table_text",
        action="store_true",
        help="Do not remove text duplicated with table regions",
    )
    return parser


def _expand_pdf_inputs(inputs: List[str]) -> List[Path]:
    pdfs: List[Path] = []
    seen: set[str] = set()
    for raw in inputs:
        path = Path(raw)
        candidates = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix.lower() != ".pdf":
                continue
            resolved = candidate.resolve()
            key = str(resolved).lower()
            if key not in seen:
                pdfs.append(resolved)
                seen.add(key)
    return pdfs


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    pdfs = _expand_pdf_inputs(args.pdf_path)
    if not pdfs:
        print("No PDF files found.")
        return 1

    failures = 0
    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {pdf_path}")
        try:
            outcome = parse_pdf_file(
                pdf_path,
                output=args.output,
                table_to_text=args.table_to_text,
                table_format=args.table_format,
                max_text_chars=args.max_text_chars,
                deduplicate_table_text=not args.no_dedup_table_text,
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED: {exc}")
            continue

        print(outcome["output_path"])
        backend = outcome["result"].get("parser_backend", "local")
        print(
            f"chunks={outcome['result']['chunk_count']} "
            f"tables={outcome['result']['table_count']} "
            f"images={outcome['result']['image_count']} "
            f"pages={outcome['result']['page_count']} "
            f"backend={backend}"
        )
        if backend == "mineru":
            print(f"mineru_markdown={outcome['result'].get('mineru_markdown', '')}")
            print(f"mineru_content_list={outcome['result'].get('mineru_content_list', '')}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
