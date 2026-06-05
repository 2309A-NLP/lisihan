# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config import Config
from src.document import Document

from .chunk_merger import merge_blocks
from .metadata_extractor import extract_document_metadata
from .mineru_backend import find_mineru_artifacts, parse_pdf_with_mineru
from .table_extractor import extract_table_blocks
from .text_extractor import extract_text_blocks
from .visual_extractor import extract_pdf_images
from .visual_geometry import _resolve_output_path


def parse_pdf_file(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    backend: str | None = None,
    table_to_text: bool = False,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    deduplicate_table_text: bool = True,
) -> Dict:
    parser_backend = (backend or Config.PDF_PARSER_BACKEND or "local").lower()
    if parser_backend == "mineru":
        return parse_pdf_with_mineru(pdf_path, output=output, max_text_chars=max_text_chars)
    if parser_backend not in {"local", "pymupdf"}:
        raise ValueError(f"Unsupported PDF parser backend: {parser_backend}")

    pdf_file = Path(pdf_path)
    text_blocks = extract_text_blocks(pdf_file)
    table_blocks = extract_table_blocks(pdf_file, table_to_text=table_to_text, table_format=table_format)
    visual_blocks = extract_pdf_images(
        pdf_file,
        Config.IMAGES_EXTRACT_DIR,
        text_blocks=text_blocks,
        table_blocks=table_blocks,
    )
    document_metadata = extract_document_metadata(pdf_file)
    merged_blocks = merge_blocks(
        text_blocks,
        table_blocks,
        table_to_text=table_to_text,
        table_format=table_format,
        max_text_chars=max_text_chars,
        deduplicate_table_text=deduplicate_table_text,
    )

    result = {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "page_count": max((block["page"] for block in merged_blocks), default=0),
        "chunk_count": len(merged_blocks),
        "table_count": len(table_blocks),
        "image_count": len(visual_blocks),
        "table_to_text": table_to_text,
        "table_format": table_format if table_to_text else "raw",
        "parser_backend": "local",
        "document_metadata": document_metadata,
        "visual_index": visual_blocks,
        "images": visual_blocks,
        "chunks": merged_blocks,
    }

    output_path = _resolve_output_path(pdf_file, output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path = output_path.parent / f"{pdf_file.stem}_metadata.json"
    metadata_path.write_text(json.dumps(document_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "output_path": str(output_path)}


class PDFParser:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, output_dir: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = output_dir or Config.PDF_PARSE_OUTPUT_DIR
        self.document_metadata: Dict[str, str] = {}
        self.document_metadata_by_company: Dict[str, Dict[str, str]] = {}

    def _to_documents(self, parsed: Dict) -> List[Document]:
        docs: List[Document] = []
        for idx, chunk in enumerate(parsed.get("chunks", [])):
            content = chunk.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            docs.append(
                Document(
                    page_content=str(content),
                    metadata={
                        **chunk.get("metadata", {}),
                        "page": chunk.get("page", 0),
                        "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                        "type": chunk.get("type", "text"),
                    },
                )
            )
        return docs

    def _load_parsed_result(self, pdf_path: str) -> Dict | None:
        pdf_file = Path(pdf_path)
        parsed_path = Path(self.output_dir) / f"{pdf_file.stem}_chunks.json"
        if not parsed_path.exists():
            return None
        try:
            return json.loads(parsed_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _ensure_mineru_artifacts(self, pdf_path: str, parsed_result: Dict) -> Dict | None:
        if (Config.PDF_PARSER_BACKEND or "").lower() != "mineru":
            return parsed_result
        if parsed_result.get("parser_backend") != "mineru":
            return None

        markdown_path = parsed_result.get("mineru_markdown")
        content_list_path = parsed_result.get("mineru_content_list")
        markdown_exists = bool(markdown_path and Path(markdown_path).exists())
        content_list_exists = bool(content_list_path and Path(content_list_path).exists())
        if not markdown_exists or not content_list_exists:
            artifacts = find_mineru_artifacts(pdf_path, self.output_dir)
            markdown = artifacts["markdown"]
            content_list = artifacts["content_list"]
            markdown_exists = bool(markdown and markdown.exists())
            content_list_exists = bool(content_list and content_list.exists())
            if markdown_exists:
                parsed_result["mineru_markdown"] = str(markdown)
            if content_list_exists:
                parsed_result["mineru_content_list"] = str(content_list)

        if not markdown_exists or not content_list_exists:
            return None
        parsed_result["parser_backend"] = "mineru"
        return parsed_result

    def _ensure_image_metadata(self, pdf_path: str, parsed_result: Dict) -> Dict:
        if "visual_index" in parsed_result and "images" in parsed_result and "image_count" in parsed_result:
            return parsed_result

        chunks = parsed_result.get("chunks", [])
        visual_blocks = extract_pdf_images(
            pdf_path,
            Config.IMAGES_EXTRACT_DIR,
            text_blocks=[chunk for chunk in chunks if chunk.get("type") == "text"],
            table_blocks=[chunk for chunk in chunks if chunk.get("type") == "table"],
        )
        parsed_result["visual_index"] = visual_blocks
        parsed_result["images"] = visual_blocks
        parsed_result["image_count"] = len(visual_blocks)

        pdf_file = Path(pdf_path)
        parsed_path = Path(self.output_dir) / f"{pdf_file.stem}_chunks.json"
        try:
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_path.write_text(json.dumps(parsed_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        return parsed_result

    def parse_pdf(self, pdf_path: str) -> List[Document]:
        parsed_result = self._load_parsed_result(pdf_path)
        if parsed_result is not None:
            parsed_result = self._ensure_mineru_artifacts(pdf_path, parsed_result)
        if parsed_result is None:
            parsed = parse_pdf_file(
                pdf_path,
                output=self.output_dir,
                backend=Config.PDF_PARSER_BACKEND,
                table_to_text=True,
                table_format="markdown",
                max_text_chars=self.chunk_size,
            )
            parsed_result = parsed["result"]
        else:
            parsed_result = self._ensure_image_metadata(pdf_path, parsed_result)
        try:
            document_metadata = extract_document_metadata(pdf_path)
            parsed_result["document_metadata"] = document_metadata
        except Exception:
            document_metadata = parsed_result.get("document_metadata", {})
        company_name = document_metadata.get("company_name")
        if company_name:
            self.document_metadata_by_company[company_name] = document_metadata
        if not self.document_metadata:
            self.document_metadata.update(document_metadata)
        elif company_name and company_name in str(pdf_path):
            self.document_metadata.update(document_metadata)
        return self._to_documents(parsed_result)

    def parse_multiple_pdfs(self, pdf_dir: str) -> List[Document]:
        pdf_root = Path(pdf_dir)
        if not pdf_root.exists():
            pdf_root.mkdir(parents=True, exist_ok=True)
            return []

        all_docs: List[Document] = []
        for pdf_file in sorted(pdf_root.glob("*.pdf")):
            all_docs.extend(self.parse_pdf(str(pdf_file)))
        return all_docs

    def extract_tables(self, pdf_path: str):
        return extract_table_blocks(pdf_path, table_to_text=False)
