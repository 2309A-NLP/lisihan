import os

import pytest

from app.utils.file_parser import FileParser


DATA_DIR = os.path.join("app", "knowledge", "data")


# Function: Find a knowledge file by token for tests.
def _find_file(token: str) -> str:
    for filename in os.listdir(DATA_DIR):
        if token in filename:
            return os.path.join(DATA_DIR, filename)
    raise FileNotFoundError(token)


# Function: Test the text pdf can be extracted behavior.
def test_text_pdf_can_be_extracted():
    pdf_path = _find_file("2025")
    text = FileParser.parse_file(pdf_path)

    assert "国家基层高血压" in text
    assert len(text) > 1000


# Function: Test the scanned course standard pdf reports ocr requirement behavior.
def test_scanned_course_standard_pdf_reports_ocr_requirement(monkeypatch):
    monkeypatch.setenv("PDF_OCR_MAX_PAGES", "1")
    pdf_path = _find_file("20220428160938")

    with pytest.raises(Exception) as exc_info:
        FileParser.parse_file(pdf_path)

    assert "扫描版PDF需要OCR" in str(exc_info.value)
    assert "Tesseract OCR" in str(exc_info.value)
