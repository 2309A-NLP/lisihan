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

from .chunk_merger import DEFAULT_CHUNK_STRATEGY, merge_blocks, normalize_chunk_strategy
from .metadata_extractor import extract_document_metadata
from .mineru_backend import has_mineru_source_files, parse_pdf_file_with_mineru
from .table_extractor import extract_table_blocks
from .text_extractor import extract_text_blocks
from .visual_extractor import extract_pdf_images
from .visual_geometry import _resolve_output_path


def _parse_pdf_file_local(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    table_to_text: bool = False,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    chunk_overlap: int = 0,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    parent_text_chars: int | None = None,
    deduplicate_table_text: bool = True,
) -> Dict:
    pdf_file = Path(pdf_path)
    chunk_strategy = normalize_chunk_strategy(chunk_strategy)
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
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        parent_text_chars=parent_text_chars,
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
        "chunk_strategy": chunk_strategy,
        "chunk_size": max_text_chars,
        "chunk_overlap": chunk_overlap,
        "parent_chunk_size": parent_text_chars,
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


def parse_pdf_file(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    table_to_text: bool = False,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    chunk_overlap: int = 0,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    parent_text_chars: int | None = None,
    deduplicate_table_text: bool = True,
) -> Dict:
    backend = getattr(Config, "PDF_PARSE_BACKEND", "mineru")
    if backend == "mineru":
        try:
            return parse_pdf_file_with_mineru(
                pdf_path,
                output=output,
                table_to_text=table_to_text,
                table_format=table_format,
                max_text_chars=max_text_chars,
                chunk_overlap=chunk_overlap,
                chunk_strategy=chunk_strategy,
                parent_text_chars=parent_text_chars,
                deduplicate_table_text=deduplicate_table_text,
            )
        except Exception:
            if not getattr(Config, "MINERU_FALLBACK_TO_LOCAL", True):
                raise
    return _parse_pdf_file_local(
        pdf_path,
        output=output,
        table_to_text=table_to_text,
        table_format=table_format,
        max_text_chars=max_text_chars,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        parent_text_chars=parent_text_chars,
        deduplicate_table_text=deduplicate_table_text,
    )


class PDFParser:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        output_dir: str = None,
        chunk_strategy: str = None,
        parent_chunk_size: int = None,
    ):
        self.chunk_strategy = normalize_chunk_strategy(chunk_strategy or getattr(Config, "PDF_CHUNK_STRATEGY", DEFAULT_CHUNK_STRATEGY))
        self.chunk_size = chunk_size if chunk_size is not None else getattr(Config, "PDF_CHUNK_SIZE", 500)
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else getattr(Config, "PDF_CHUNK_OVERLAP", 50)
        self.parent_chunk_size = parent_chunk_size if parent_chunk_size is not None else getattr(
            Config,
            "PDF_PARENT_CHUNK_SIZE",
            max(self.chunk_size * 3, self.chunk_size),
        )
        self.output_dir = output_dir or Config.PDF_PARSE_OUTPUT_DIR
        self.document_metadata: Dict[str, str] = {}
        self.document_metadata_by_company: Dict[str, Dict[str, str]] = {}

    def _to_documents(self, parsed: Dict) -> List[Document]:
        docs: List[Document] = []
        source_file = parsed.get("source_file", "")
        source_path = parsed.get("source_path", "")
        for idx, chunk in enumerate(parsed.get("chunks", [])):
            content = chunk.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            metadata = dict(chunk.get("metadata", {}) or {})
            if source_file and "source_file" not in metadata:
                metadata["source_file"] = source_file
            if source_path and "source_path" not in metadata:
                metadata["source_path"] = source_path
            docs.append(
                Document(
                    page_content=str(content),
                    metadata={
                        **metadata,
                        "page": chunk.get("page", 0),
                        "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                        "type": chunk.get("type", "text"),
                    },
                )
            )
        return docs

    def _is_cached_result_compatible(self, parsed_result: Dict) -> bool:
        cached_strategy = normalize_chunk_strategy(parsed_result.get("chunk_strategy"))
        if "chunk_strategy" not in parsed_result and self.chunk_strategy == "paragraph":
            return True
        cached_backend = parsed_result.get("parser_backend", "local")
        expected_backend = getattr(Config, "PDF_PARSE_BACKEND", "mineru")
        if cached_backend != expected_backend:
            if (
                expected_backend == "mineru"
                and cached_backend == "local"
                and getattr(Config, "MINERU_FALLBACK_TO_LOCAL", True)
                and not has_mineru_source_files(parsed_result.get("source_path", ""))
            ):
                pass
            else:
                return False
        return (
            cached_strategy == self.chunk_strategy
            and int(parsed_result.get("chunk_size", self.chunk_size) or self.chunk_size) == int(self.chunk_size)
            and int(parsed_result.get("chunk_overlap", self.chunk_overlap) or 0) == int(self.chunk_overlap)
            and int(parsed_result.get("parent_chunk_size", self.parent_chunk_size) or self.parent_chunk_size)
            == int(self.parent_chunk_size)
        )

    def _load_parsed_result(self, pdf_path: str) -> Dict | None:
        pdf_file = Path(pdf_path)
        parsed_path = Path(self.output_dir) / f"{pdf_file.stem}_chunks.json"
        if not parsed_path.exists():
            return None
        try:
            parsed_result = json.loads(parsed_path.read_text(encoding="utf-8"))
            return parsed_result if self._is_cached_result_compatible(parsed_result) else None
        except Exception:
            return None

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
        if parsed_result is None:
            parsed = parse_pdf_file(
                pdf_path,
                output=self.output_dir,
                table_to_text=True,
                table_format="markdown",
                max_text_chars=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunk_strategy=self.chunk_strategy,
                parent_text_chars=self.parent_chunk_size,
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
