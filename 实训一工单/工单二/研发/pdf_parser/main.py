# -*- coding: utf-8 -*-
"""Use MinerU outputs as the single PDF parsing source."""

from __future__ import annotations

import argparse
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

from src.config import Config
from src.document import Document
from utils.logger import get_logger

from .chunk_merger import merge_blocks
from .metadata_extractor import extract_document_metadata
from .mineru_wsl import (
    extract_blocks_with_mineru,
    find_existing_content_list,
    load_mineru_blocks,
)


logger = get_logger(__name__)


class _TableTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] | None = None
        self._current_cell: List[str] | None = None

    def handle_starttag(self, tag: str, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []

    def handle_data(self, data: str):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str):
        if tag in {"td", "th"} and self._current_cell is not None:
            cell = self._clean_cell("".join(self._current_cell))
            if self._current_row is not None and cell:
                self._current_row.append(cell)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None

    @staticmethod
    def _clean_cell(text: str) -> str:
        return re.sub(r"\s+", " ", unescape(text or "")).strip()

    def as_text(self) -> str:
        return "\n".join(" | ".join(row) for row in self.rows)


def _output_root(output: str | None = None) -> Path:
    root = Path(output or Config.PDF_PARSE_OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _find_markdown(root: Path, pdf_stem: str) -> Path | None:
    candidates = list(root.rglob(f"{pdf_stem}.md"))
    if not candidates:
        candidates = list(root.rglob("*.md"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _normalize_inline_text(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text or "")).strip()


def _escape_markdown_table_cell(text: str) -> str:
    return _normalize_inline_text(text).replace("|", "\\|")


def _html_table_to_markdown(table) -> str:
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [_escape_markdown_table_cell(cell.get_text(" ", strip=True)) for cell in cells]
        row = [cell for cell in row if cell]
        if row:
            rows.append(row)

    if not rows:
        plain_text = _normalize_inline_text(table.get_text(" ", strip=True))
        return plain_text

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = padded_rows[0]
    separator = ["---"] * width
    body = padded_rows[1:]

    markdown_rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_rows.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(markdown_rows)


def _replace_html_tables_with_markdown(text: str) -> tuple[str, int]:
    """Convert only HTML table nodes to Markdown, preserving surrounding text."""
    soup = BeautifulSoup(text or "", "html.parser")
    tables = soup.find_all("table")
    for table in tables:
        table.replace_with("\n" + _html_table_to_markdown(table) + "\n")
    return str(soup), len(tables)


def _log_cleaning_diagnostics(original: str, cleaned: str, table_count: int) -> None:
    keywords = ["下游", "行业", "上游", "终端用户", "军队、政府机关、能源"]
    before_hits = {keyword: keyword in original for keyword in keywords}
    after_hits = {keyword: keyword in cleaned for keyword in keywords}
    missing_after = [keyword for keyword in keywords if before_hits[keyword] and not after_hits[keyword]]

    logger.info(
        "mineru markdown cleaned | before_chars=%s | after_chars=%s | tables=%s | keyword_hits_before=%s | keyword_hits_after=%s",
        len(original or ""),
        len(cleaned or ""),
        table_count,
        before_hits,
        after_hits,
    )
    if missing_after:
        logger.warning("mineru markdown cleaning lost keywords | keywords=%s", missing_after)


def _clean_mineru_markdown(text: str) -> str:
    original = text or ""
    text, table_count = _replace_html_tables_with_markdown(original)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>|</div>", "\n", text, flags=re.IGNORECASE)
    text = BeautifulSoup(text, "html.parser").get_text("\n")
    text = unescape(text)
    text = re.sub(r"\b(?:rens?|Tengon(?:g|e|y|z)?|chinene|cengon|ong|gong)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = text.strip()
    _log_cleaning_diagnostics(original, cleaned, table_count)
    return cleaned


def _is_markdown_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S+", line.strip()))


def _heading_level(line: str) -> int:
    match = re.match(r"^(#{1,6})\s+", line.strip())
    return len(match.group(1)) if match else 0


def _split_markdown_sections(text: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    heading_stack: List[tuple[int, str]] = []
    current_lines: List[str] = []
    current_title = "全文"

    def flush() -> None:
        nonlocal current_lines, current_title
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"title": current_title, "content": body})
        current_lines = []

    for line in (text or "").splitlines():
        stripped = line.strip()
        if _is_markdown_heading(stripped):
            flush()
            level = _heading_level(stripped)
            title_text = re.sub(r"^#{1,6}\s+", "", stripped).strip()
            heading_stack[:] = [(item_level, item_title) for item_level, item_title in heading_stack if item_level < level]
            heading_stack.append((level, title_text))
            current_title = " > ".join(item_title for _, item_title in heading_stack)
            current_lines.append(stripped)
        else:
            current_lines.append(line)
    flush()
    return sections


def _split_long_text_recursively(text: str, chunk_size: int) -> List[str]:
    text = (text or "").strip()
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    separators = ["\n\n", "\n", "。", "；", ";", "，", ",", " "]
    for separator in separators:
        if separator not in text:
            continue
        pieces = text.split(separator)
        chunks: List[str] = []
        current = ""
        for index, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            suffix = separator if separator in {"。", "；", ";", "，", ","} and index < len(pieces) - 1 else ""
            candidate_piece = piece + suffix
            candidate = f"{current}{separator if current and separator not in {'。', '；', ';', '，', ',', ' '} else ''}{candidate_piece}".strip()
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.extend(_split_long_text_recursively(current, chunk_size))
            current = candidate_piece
        if current:
            chunks.extend(_split_long_text_recursively(current, chunk_size))
        if chunks:
            return chunks

    return [text[index : index + chunk_size].strip() for index in range(0, len(text), chunk_size) if text[index : index + chunk_size].strip()]


def _split_section_to_child_chunks(section_text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", section_text or "") if block.strip()]
    chunks: List[str] = []
    current = ""

    def add_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        if len(block) > chunk_size:
            add_current()
            chunks.extend(_split_long_text_recursively(block, chunk_size))
            continue
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) > chunk_size:
            add_current()
            current = block
        else:
            current = candidate
    add_current()

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: List[str] = []
    previous_tail = ""
    for chunk in chunks:
        if previous_tail:
            combined = f"{previous_tail}\n\n{chunk}".strip()
            overlapped.append(combined[-chunk_size:] if len(combined) > chunk_size else combined)
        else:
            overlapped.append(chunk)
        previous_tail = chunk[-chunk_overlap:]
    return overlapped


def _limit_parent_context(title: str, content: str, max_chars: int = 2500) -> str:
    parent = f"{title}\n\n{content}".strip() if title else (content or "").strip()
    if len(parent) <= max_chars:
        return parent
    return parent[:max_chars].rstrip() + "\n...[父块已截断]"


def _chunk_markdown_text(text: str, chunk_size: int, chunk_overlap: int = 120) -> List[str]:
    """Backward-compatible helper returning child chunk text only."""
    return [item["content"] for item in _chunk_markdown_hierarchical(text, chunk_size, chunk_overlap)]


def _chunk_markdown_hierarchical(text: str, chunk_size: int, chunk_overlap: int = 120) -> List[Dict[str, str]]:
    child_chunks: List[Dict[str, str]] = []
    sections = _split_markdown_sections(text)
    for parent_index, section in enumerate(sections, start=1):
        title = section["title"]
        content = section["content"]
        parent_id = f"parent_{parent_index}"
        parent_context = _limit_parent_context(title, content)
        for child_index, child in enumerate(_split_section_to_child_chunks(content, chunk_size, chunk_overlap), start=1):
            child_text = f"{title}\n\n{child}".strip() if title and title not in child[:80] else child
            child_chunks.append(
                {
                    "content": child_text,
                    "parent_id": parent_id,
                    "parent_title": title,
                    "parent_content": parent_context,
                    "child_index": str(child_index),
                }
            )
    logger.info("mineru markdown chunked | sections=%s | child_chunks=%s | strategy=heading+paragraph+recursive+parent_child", len(sections), len(child_chunks))
    return child_chunks


def _build_result(
    pdf_file: Path,
    *,
    text_blocks: List[Dict],
    table_blocks: List[Dict],
    mineru_content_list: str,
    mineru_markdown: str = "",
    mineru_raw_output_dir: str = "",
    table_to_text: bool = True,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    deduplicate_table_text: bool = True,
) -> Dict:
    document_metadata = extract_document_metadata(pdf_file)
    merged_blocks = merge_blocks(
        text_blocks,
        table_blocks,
        table_to_text=table_to_text,
        table_format=table_format,
        max_text_chars=max_text_chars,
        deduplicate_table_text=deduplicate_table_text,
    )
    return {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "parser_backend": "mineru",
        "page_count": max((block["page"] for block in merged_blocks), default=0),
        "chunk_count": len(merged_blocks),
        "table_count": len(table_blocks),
        "table_to_text": table_to_text,
        "table_format": table_format if table_to_text else "raw",
        "document_metadata": document_metadata,
        "mineru_content_list": mineru_content_list,
        "mineru_markdown": mineru_markdown,
        "mineru_raw_output_dir": mineru_raw_output_dir,
        "chunks": merged_blocks,
    }


def _documents_from_markdown(markdown_path: Path, pdf_file: Path, *, chunk_size: int, chunk_overlap: int = 120) -> List[Document]:
    text = _clean_mineru_markdown(markdown_path.read_text(encoding="utf-8", errors="replace"))
    docs: List[Document] = []
    for idx, chunk in enumerate(_chunk_markdown_hierarchical(text, chunk_size, chunk_overlap), start=1):
        docs.append(
            Document(
                page_content=chunk["content"],
                metadata={
                    "source_file": pdf_file.name,
                    "page": 0,
                    "chunk_id": f"mineru_{idx}",
                    "source_path": str(pdf_file),
                    "markdown_path": str(markdown_path),
                    "parser_backend": "mineru",
                    "chunk_source": "mineru_markdown_child",
                    "chunk_strategy": "heading_paragraph_recursive_parent_child",
                    "parent_id": chunk["parent_id"],
                    "parent_title": chunk["parent_title"],
                    "parent_content": chunk["parent_content"],
                    "child_index": chunk["child_index"],
                },
            )
        )
    return docs


def load_parsed_pdf_file(pdf_path: str | Path, *, output: str | None = None) -> Dict | None:
    pdf_file = Path(pdf_path)
    root = _output_root(output)

    chunks_path = root / f"{pdf_file.stem}_chunks.json"
    if chunks_path.exists():
        parsed = json.loads(chunks_path.read_text(encoding="utf-8"))
        if parsed.get("parser_backend") == "mineru":
            return {"result": parsed, "output_path": str(chunks_path)}

    content_list_path = find_existing_content_list(root, pdf_file.stem)
    if content_list_path is None:
        return None

    blocks = load_mineru_blocks(content_list_path, pdf_file)
    table_blocks = [block for block in blocks if block.get("type") == "table"]
    text_blocks = [block for block in blocks if block.get("type") != "table"]
    markdown_candidates = sorted(root.rglob(f"{pdf_file.stem}.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    result = _build_result(
        pdf_file,
        text_blocks=text_blocks,
        table_blocks=table_blocks,
        mineru_content_list=str(content_list_path),
        mineru_markdown=str(markdown_candidates[0]) if markdown_candidates else "",
        mineru_raw_output_dir=str(content_list_path.parent.parent),
    )
    return {"result": result, "output_path": str(content_list_path)}


def parse_pdf_file(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    parser_backend: str = "mineru",
    table_to_text: bool = True,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    deduplicate_table_text: bool = True,
) -> Dict:
    if parser_backend.lower().strip() != "mineru":
        raise ValueError("Only MinerU parsing is supported. Set parser_backend='mineru'.")

    pdf_file = Path(pdf_path)
    root = _output_root(output)
    mineru_result = extract_blocks_with_mineru(pdf_file, output_root=root)
    result = _build_result(
        pdf_file,
        text_blocks=mineru_result["text_blocks"],
        table_blocks=mineru_result["table_blocks"],
        mineru_content_list=mineru_result["content_list_path"],
        mineru_markdown=mineru_result["exported_markdown_path"] or mineru_result["markdown_path"],
        mineru_raw_output_dir=mineru_result["raw_output_dir"],
        table_to_text=table_to_text,
        table_format=table_format,
        max_text_chars=max_text_chars,
        deduplicate_table_text=deduplicate_table_text,
    )
    return {"result": result, "output_path": result["mineru_content_list"]}


class PDFParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 120, output_dir: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = output_dir or Config.PDF_PARSE_OUTPUT_DIR
        self.document_metadata: Dict[str, str] = {}

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
                        "parser_backend": "mineru",
                    },
                )
            )
        return docs

    def _parse_pdf(self, pdf_path: str, *, force_reparse: bool = False) -> List[Document]:
        parsed = None if force_reparse else load_parsed_pdf_file(pdf_path, output=self.output_dir)
        if parsed is None:
            parsed = parse_pdf_file(
                pdf_path,
                output=self.output_dir,
                parser_backend="mineru",
                table_to_text=True,
                table_format="markdown",
                max_text_chars=self.chunk_size,
            )
        self.document_metadata.update(parsed["result"].get("document_metadata", {}))
        return self._to_documents(parsed["result"])

    def parse_pdf(self, pdf_path: str) -> List[Document]:
        return self._parse_pdf(pdf_path, force_reparse=True)

    def load_parsed_pdf(self, pdf_path: str) -> List[Document]:
        pdf_file = Path(pdf_path)
        markdown_path = _find_markdown(Path(self.output_dir), pdf_file.stem)
        if markdown_path is None:
            return []
        try:
            self.document_metadata.update(extract_document_metadata(pdf_file))
        except Exception:
            pass
        return _documents_from_markdown(markdown_path, pdf_file, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def _parse_multiple_pdfs(self, pdf_dir: str, *, force_reparse: bool) -> List[Document]:
        pdf_root = Path(pdf_dir)
        if not pdf_root.exists():
            pdf_root.mkdir(parents=True, exist_ok=True)
            return []

        all_docs: List[Document] = []
        for pdf_file in sorted(pdf_root.glob("*.pdf")):
            all_docs.extend(self._parse_pdf(str(pdf_file), force_reparse=force_reparse))
        return all_docs

    def parse_multiple_pdfs(self, pdf_dir: str) -> List[Document]:
        return self._parse_multiple_pdfs(pdf_dir, force_reparse=True)

    def load_parsed_multiple_pdfs(self, pdf_dir: str) -> List[Document]:
        pdf_root = Path(pdf_dir)
        if not pdf_root.exists():
            return []

        all_docs: List[Document] = []
        for pdf_file in sorted(pdf_root.glob("*.pdf")):
            all_docs.extend(self.load_parsed_pdf(str(pdf_file)))
        return all_docs

    def reparse_multiple_pdfs(self, pdf_dir: str) -> List[Document]:
        return self._parse_multiple_pdfs(pdf_dir, force_reparse=True)

    def load_or_parse_multiple_pdfs(self, pdf_dir: str) -> List[Document]:
        docs = self.load_parsed_multiple_pdfs(pdf_dir)
        return docs if docs else self.reparse_multiple_pdfs(pdf_dir)

    def extract_tables(self, pdf_path: str):
        parsed = load_parsed_pdf_file(pdf_path, output=self.output_dir) or parse_pdf_file(
            pdf_path,
            output=self.output_dir,
            parser_backend="mineru",
        )
        return [chunk for chunk in parsed["result"].get("chunks", []) if chunk.get("type") == "table"]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF招股说明书 MinerU 解析脚本")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("-o", "--output", default=None, help="MinerU 输出目录，默认 mineru_output")
    parser.add_argument("--table_to_text", action="store_true", default=True, help="将表格转换为可读文本")
    parser.add_argument("--table_format", choices=["markdown", "csv"], default="markdown", help="表格转文本格式")
    parser.add_argument("--max_text_chars", type=int, default=1500, help="文本块最大字符数")
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
    )
    print(outcome["output_path"])
    print(
        f"chunks={outcome['result']['chunk_count']} "
        f"tables={outcome['result']['table_count']} "
        f"pages={outcome['result']['page_count']} "
        f"backend=mineru"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
