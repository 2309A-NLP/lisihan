# -*- coding: utf-8 -*-
"""Run MinerU for one or more PDFs and adapt outputs to *_chunks.json."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from pdf_parser.mineru_backend import parse_pdf_with_mineru
from src.config import Config


def _expand_inputs(paths: Iterable[str]) -> List[Path]:
    pdfs: List[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            pdfs.append(path)
    seen = set()
    unique: List[Path] = []
    for pdf in pdfs:
        key = str(pdf.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(pdf)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MinerU through WSL and write project chunks JSON")
    parser.add_argument("inputs", nargs="+", help="PDF files or directories containing PDFs")
    parser.add_argument("--output-dir", default=Config.MINERU_OUTPUT_DIR, help="MinerU/project output directory")
    parser.add_argument("--max-text-chars", type=int, default=1500, help="Maximum markdown text per chunk")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    pdfs = _expand_inputs(args.inputs)
    if not pdfs:
        parser.error("No PDF files found.")

    for pdf in pdfs:
        outcome = parse_pdf_with_mineru(pdf, output=args.output_dir, max_text_chars=args.max_text_chars)
        result = outcome["result"]
        print(f"[OK] {pdf}")
        print(f"  output={outcome['output_path']}")
        print(f"  chunks={result.get('chunk_count', 0)} tables={result.get('table_count', 0)} images={result.get('image_count', 0)} pages={result.get('page_count', 0)}")
        print(f"  mineru_markdown={result.get('mineru_markdown', '')}")
        print(f"  mineru_content_list={result.get('mineru_content_list', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

