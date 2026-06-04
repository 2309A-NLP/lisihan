# -*- coding: utf-8 -*-
"""PDF parsing module."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List

from src.document import Document
from src.mineru_wsl import parse_pdf_with_mineru_wsl
from utils.logger import get_logger


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
        text = re.sub(r"\s+", " ", unescape(text or "")).strip()
        return text

    def as_text(self) -> str:
        return "\n".join(" | ".join(row) for row in self.rows)


class PDFParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def _chunk_markdown_text(self, text: str) -> List[str]:
        blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
        chunks: List[str] = []
        current = ""
        for block in blocks:
            if len(block) >= self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(block)
                continue
            candidate = f"{current}\n\n{block}".strip() if current else block
            if len(candidate) > self.chunk_size and current:
                chunks.append(current.strip())
                current = block
            else:
                current = candidate
        if current:
            chunks.append(current.strip())
        return chunks

    def _normalize_text(self, text: str) -> str:
        text = self._clean_mineru_markdown(text or "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_mineru_markdown(self, text: str) -> str:
        def replace_table(match: re.Match) -> str:
            parser = _TableTextParser()
            parser.feed(match.group(0))
            return "\n" + parser.as_text() + "\n"

        text = re.sub(r"<table.*?</table>", replace_table, text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>|</div>|</tr>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\b(?:rens?|Tengon(?:g|e|y|z)?|chinene|cengon|ong|gong)\b", " ", text, flags=re.IGNORECASE)
        return text

    def _documents_from_markdown(self, markdown_path: Path, pdf_file: Path) -> List[Document]:
        text = self._normalize_text(markdown_path.read_text(encoding="utf-8", errors="replace"))
        chunks: List[Document] = []
        for idx, chunk in enumerate(self._chunk_markdown_text(text)):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source_file": pdf_file.name,
                        "page": 0,
                        "chunk_id": f"mineru_{idx}",
                        "source_path": str(pdf_file),
                        "markdown_path": str(markdown_path),
                        "parser": "mineru",
                    },
                )
            )
        return chunks

    def _parse_pdf_with_mineru(self, pdf_file: Path) -> List[Document]:
        result = parse_pdf_with_mineru_wsl(pdf_file)
        chunks = self._documents_from_markdown(result.markdown_path, pdf_file)
        logger.info(
            "mineru markdown parsed | file=%s | markdown=%s | chunks=%s | cached=%s",
            pdf_file,
            result.markdown_path,
            len(chunks),
            result.cached,
        )
        return chunks

    def parse_pdf(self, pdf_path: str) -> List[Document]:
        pdf_file = Path(pdf_path)
        try:
            logger.info("parse_pdf start | file=%s", pdf_file)
            chunks = self._parse_pdf_with_mineru(pdf_file)
            logger.info("parse_pdf done | file=%s | chunks=%s | parser=mineru", pdf_file, len(chunks))
            return chunks
        except Exception:
            logger.exception("parse_pdf failed | file=%s", pdf_file)
            return []

    def parse_multiple_pdfs(self, pdf_dir: str) -> List[Document]:
        all_chunks: List[Document] = []
        pdf_root = Path(pdf_dir)
        if not pdf_root.exists():
            pdf_root.mkdir(parents=True, exist_ok=True)
            logger.warning("pdf dir created | dir=%s", pdf_root)
            return all_chunks

        for pdf_file in sorted(pdf_root.glob("*.pdf")):
            chunks = self.parse_pdf(str(pdf_file))
            all_chunks.extend(chunks)
            logger.info("pdf parsed | file=%s | chunks=%s", pdf_file.name, len(chunks))

        logger.info("parse_multiple_pdfs done | dir=%s | total_chunks=%s", pdf_root, len(all_chunks))
        return all_chunks

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        logger.info("extract_tables called | file=%s", pdf_path)
        return []

