# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import argparse
import json
from typing import List

from .parser import parse_pdf_file


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF prospectus parsing and chunking")
    parser.add_argument("pdf_path", help="PDF file path")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file or directory")
    parser.add_argument(
        "--backend",
        choices=["mineru", "local", "pymupdf"],
        default=None,
        help="Parser backend; defaults to Config.PDF_PARSER_BACKEND",
    )
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


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    outcome = parse_pdf_file(
        args.pdf_path,
        output=args.output,
        backend=args.backend,
        table_to_text=args.table_to_text,
        table_format=args.table_format,
        max_text_chars=args.max_text_chars,
        deduplicate_table_text=not args.no_dedup_table_text,
    )
    result = outcome["result"]
    print(f"output={outcome['output_path']}")
    print(f"chunks={result.get('chunk_count', 0)}")
    print(f"tables={result.get('table_count', 0)}")
    print(f"images={result.get('image_count', 0)}")
    print(f"pages={result.get('page_count', 0)}")
    print(f"parser_backend={result.get('parser_backend', '')}")
    print(f"mineru_markdown={result.get('mineru_markdown', '')}")
    print(f"mineru_content_list={result.get('mineru_content_list', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
