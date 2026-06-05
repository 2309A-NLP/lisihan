# -*- coding: utf-8 -*-
"""Example: parse a PDF with WSL MinerU and convert it to RAG chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdf_parser.main import PDFParser, parse_pdf_file_with_mineru_wsl


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a PDF with WSL MinerU.")
    parser.add_argument("pdf_path", help="PDF path on Windows, for example data\\file.pdf.")
    parser.add_argument("-o", "--output", default="parsed_pdfs", help="Chunk JSON output directory.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Max chunk characters.")
    parser.add_argument("--as-documents", action="store_true", help="Also load chunks as RAG Document objects.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    parsed = parse_pdf_file_with_mineru_wsl(
        args.pdf_path,
        output=args.output,
        max_text_chars=args.chunk_size,
    )
    result = parsed["result"]
    print(parsed["output_path"])
    print(
        f"chunks={result['chunk_count']} "
        f"tables={result['table_count']} "
        f"pages={result['page_count']} "
        f"parser={result['parser']}"
    )

    if args.as_documents:
        docs = PDFParser(chunk_size=args.chunk_size, output_dir=args.output).parse_pdf(args.pdf_path)
        print(f"documents={len(docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
