# -*- coding: utf-8 -*-
"""PyCharm example: call MinerU installed in WSL to parse a PDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mineru_wsl import parse_pdf_with_mineru_wsl


def main() -> None:
    parser = argparse.ArgumentParser(description="Use WSL MinerU to parse a PDF from Windows Python.")
    parser.add_argument("pdf", nargs="?", default="data/招股说明书1.pdf", help="PDF path on Windows.")
    parser.add_argument("-o", "--output", default="mineru_output", help="Output directory on Windows.")
    parser.add_argument("-m", "--method", default="auto", help="MinerU parse method, usually auto/ocr/txt.")
    parser.add_argument("--no-cache", action="store_true", help="Run MinerU even if an existing Markdown result is found.")
    args = parser.parse_args()

    result = parse_pdf_with_mineru_wsl(
        pdf_path=Path(args.pdf),
        output_dir=Path(args.output),
        method=args.method,
        use_cache=not args.no_cache,
    )

    print("MinerU 解析完成")
    print(f"PDF: {result.pdf_path}")
    print(f"输出目录: {result.output_dir}")
    print(f"Markdown: {result.markdown_path}")
    print(f"使用缓存: {result.cached}")
    if result.command:
        print("命令:")
        print(" ".join(result.command))


if __name__ == "__main__":
    main()
