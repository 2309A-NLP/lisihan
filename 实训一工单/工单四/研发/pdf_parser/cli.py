# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import argparse
from typing import List

from .parser import parse_pdf_file


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF prospectus parsing and chunking")
    parser.add_argument("pdf_path", help="PDF file path")
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
        "--chunk_overlap",
        type=int,
        default=0,
        help="Character overlap for recursive, heading, and parent-child chunks",
    )
    parser.add_argument(
        "--parent_text_chars",
        type=int,
        default=None,
        help="Maximum parent chunk size for parent-child chunking",
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
        table_to_text=args.table_to_text,
        table_format=args.table_format,
        max_text_chars=args.max_text_chars,
        chunk_strategy="combined",
        chunk_overlap=args.chunk_overlap,
        parent_text_chars=args.parent_text_chars,
        deduplicate_table_text=not args.no_dedup_table_text,
    )
    print(outcome["output_path"])
    print(
        f"chunks={outcome['result']['chunk_count']} "
        f"tables={outcome['result']['table_count']} "
        f"visuals={outcome['result']['image_count']} "
        f"pages={outcome['result']['page_count']} "
        f"backend={outcome['result'].get('parser_backend', 'unknown')}"
    )
    if outcome["result"].get("mineru_markdown"):
        print(f"markdown={outcome['result']['mineru_markdown']}")
    if outcome["result"].get("mineru_content_list"):
        print(f"content_list={outcome['result']['mineru_content_list']}")
    return 0
