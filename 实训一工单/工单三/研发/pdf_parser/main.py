# -*- coding: utf-8 -*-
"""CLI entry point for PDF parsing and chunking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.config import Config
from src.document import Document

from .chunk_merger import merge_blocks
from .chunk_strategies import CHUNK_STRATEGIES, normalize_chunk_strategy
from .mineru_converter import mineru_output_to_parsed_result
from .mineru_wsl import parse_pdf_with_mineru_wsl
from .metadata_extractor import extract_document_metadata
from .table_extractor import extract_table_blocks
from .text_extractor import extract_text_blocks


def _resolve_output_path(pdf_path: Path, output: str | None) -> Path:
    default_dir = Path("parsed_output")
    if not output:
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"{pdf_path.stem}_chunks.json"

    output_path = Path(output)
    if output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{pdf_path.stem}_chunks.json"


def parse_pdf_file(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    table_to_text: bool = False,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    chunk_overlap: int = 0,
    chunk_strategy: str = "paragraph",
    deduplicate_table_text: bool = True,
) -> Dict:
    pdf_file = Path(pdf_path)
    normalized_strategy = normalize_chunk_strategy(chunk_strategy)
    text_blocks = extract_text_blocks(pdf_file)
    table_blocks = extract_table_blocks(pdf_file, table_to_text=table_to_text, table_format=table_format)
    document_metadata = extract_document_metadata(pdf_file)
    merged_blocks = merge_blocks(
        text_blocks,
        table_blocks,
        table_to_text=table_to_text,
        table_format=table_format,
        max_text_chars=max_text_chars,
        chunk_overlap=chunk_overlap,
        chunk_strategy=normalized_strategy,
        deduplicate_table_text=deduplicate_table_text,
    )

    result = {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "page_count": max((block["page"] for block in merged_blocks), default=0),
        "chunk_count": len(merged_blocks),
        "table_count": len(table_blocks),
        "table_to_text": table_to_text,
        "table_format": table_format if table_to_text else "raw",
        "chunk_strategy": normalized_strategy,
        "chunk_size": max_text_chars,
        "chunk_overlap": chunk_overlap,
        "document_metadata": document_metadata,
        "chunks": merged_blocks,
    }

    output_path = _resolve_output_path(pdf_file, output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path = output_path.parent / f"{pdf_file.stem}_metadata.json"
    metadata_path.write_text(json.dumps(document_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "output_path": str(output_path)}


def parse_pdf_file_with_mineru_wsl(
    pdf_path: str | Path,
    *,
    output: str | Path | None = None,
    max_text_chars: int = 1500,
    chunk_overlap: int = 0,
    chunk_strategy: str = "paragraph",
) -> Dict:
    mineru_result = parse_pdf_with_mineru_wsl(
        pdf_path,
        output_dir=Config.MINERU_OUTPUT_DIR,
        conda_prefix=Config.MINERU_WSL_CONDA_PREFIX,
        conda_env=Config.MINERU_WSL_CONDA_ENV,
        wsl_distro=Config.MINERU_WSL_DISTRO,
        method=Config.MINERU_METHOD,
        timeout=Config.MINERU_TIMEOUT,
        use_cache=Config.MINERU_USE_CACHE,
    )
    return mineru_output_to_parsed_result(
        pdf_path,
        markdown_path=mineru_result.markdown_path,
        content_list_path=mineru_result.content_list_path,
        output=output or Config.PDF_PARSE_OUTPUT_DIR,
        max_text_chars=max_text_chars,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
    )


class PDFParser:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        output_dir: str = None,
        chunk_strategy: str = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = output_dir or Config.PDF_PARSE_OUTPUT_DIR
        self.chunk_strategy = normalize_chunk_strategy(chunk_strategy or getattr(Config, "PDF_CHUNK_STRATEGY", "paragraph"))
        self.document_metadata: Dict[str, str] = {}
        self.document_metadata_by_company: Dict[str, Dict[str, str]] = {}
        self.last_mineru_error = ""

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

    def _is_mineru_parsed_result(self, parsed_result: Dict | None) -> bool:
        if not parsed_result:
            return False
        if parsed_result.get("parser") == "mineru":
            return True
        chunks = parsed_result.get("chunks") or []
        if not chunks:
            return False
        first_metadata = chunks[0].get("metadata", {}) if isinstance(chunks[0], dict) else {}
        return first_metadata.get("parser") == "mineru"

    def _matches_chunk_options(self, parsed_result: Dict | None) -> bool:
        if not parsed_result:
            return False
        return (
            normalize_chunk_strategy(parsed_result.get("chunk_strategy")) == self.chunk_strategy
            and int(parsed_result.get("chunk_size", 0) or 0) == int(self.chunk_size)
            and int(parsed_result.get("chunk_overlap", 0) or 0) == int(self.chunk_overlap)
        )

    def _parse_with_mineru(self, pdf_path: str) -> Dict | None:
        if not Config.ENABLE_MINERU_WSL:
            return None
        try:
            parsed = parse_pdf_file_with_mineru_wsl(
                pdf_path,
                output=self.output_dir,
                max_text_chars=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunk_strategy=self.chunk_strategy,
            )
            self.last_mineru_error = ""
            return parsed["result"]
        except Exception as exc:
            self.last_mineru_error = str(exc)
            return None

    def parse_pdf(self, pdf_path: str) -> List[Document]:
        parsed_result = self._load_parsed_result(pdf_path)
        if parsed_result is not None and not self._matches_chunk_options(parsed_result):
            parsed_result = None
        if Config.ENABLE_MINERU_WSL and not self._is_mineru_parsed_result(parsed_result):
            mineru_parsed = self._parse_with_mineru(pdf_path)
            if mineru_parsed is not None:
                parsed_result = mineru_parsed

        if parsed_result is None:
            parsed = parse_pdf_file(
                pdf_path,
                output=self.output_dir,
                table_to_text=True,
                table_format="markdown",
                max_text_chars=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunk_strategy=self.chunk_strategy,
            )
            parsed_result = parsed["result"]
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF招股说明书智能解析与分块脚本")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("--mineru-wsl", action="store_true", help="Use MinerU installed in WSL.")
    parser.add_argument("--mineru-output", default="mineru_output", help="MinerU output directory.")
    parser.add_argument("--mineru-conda-prefix", default="/home/li/miniconda3", help="Conda prefix in WSL.")
    parser.add_argument("--mineru-conda-env", default="base", help="Conda env name in WSL.")
    parser.add_argument("--mineru-wsl-distro", default="", help="Optional WSL distribution name.")
    parser.add_argument("--mineru-method", default="auto", choices=["auto", "ocr", "txt"], help="MinerU parse method.")
    parser.add_argument("-o", "--output", default=None, help="输出JSON文件或目录")
    parser.add_argument("--table_to_text", action="store_true", help="将表格转换为可读文本")
    parser.add_argument(
        "--table_format",
        choices=["markdown", "csv"],
        default="markdown",
        help="表格转文本时的格式",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=1500,
        help="文本块最大字符数，超过后按句子继续切分",
    )
    parser.add_argument(
        "--no_dedup_table_text",
        action="store_true",
        help="不移除与表格区域重叠的文本块",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Adjacent chunk overlap characters for recursive/title/parent-child chunking.",
    )
    parser.add_argument(
        "--chunk_strategy",
        choices=list(CHUNK_STRATEGIES),
        default="paragraph",
        help="Chunking strategy: paragraph, recursive, title, or parent_child.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.mineru_wsl:
        from dataclasses import asdict

        try:
            from .mineru_wsl import parse_pdf_with_mineru_wsl
        except ImportError:  # pragma: no cover - supports running as a script
            from pdf_parser.mineru_wsl import parse_pdf_with_mineru_wsl

        result = parse_pdf_with_mineru_wsl(
            args.pdf_path,
            output_dir=args.mineru_output,
            conda_prefix=args.mineru_conda_prefix,
            conda_env=args.mineru_conda_env,
            wsl_distro=args.mineru_wsl_distro,
            method=args.mineru_method,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0

    outcome = parse_pdf_file(
        args.pdf_path,
        output=args.output,
        table_to_text=args.table_to_text,
        table_format=args.table_format,
        max_text_chars=args.max_text_chars,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=args.chunk_strategy,
        deduplicate_table_text=not args.no_dedup_table_text,
    )
    print(outcome["output_path"])
    print(
        f"chunks={outcome['result']['chunk_count']} "
        f"tables={outcome['result']['table_count']} "
        f"pages={outcome['result']['page_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
