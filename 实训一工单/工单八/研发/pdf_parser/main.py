# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from .cli import build_argument_parser, main
from .parser import PDFParser, parse_pdf_file
from .visual_extractor import extract_pdf_images, extract_pdf_visuals, render_pdf_page_to_image

__all__ = [
    "PDFParser",
    "parse_pdf_file",
    "extract_pdf_images",
    "extract_pdf_visuals",
    "render_pdf_page_to_image",
    "build_argument_parser",
    "main",
]


if __name__ == "__main__":
    main()
