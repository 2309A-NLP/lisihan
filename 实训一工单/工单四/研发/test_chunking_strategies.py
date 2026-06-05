from pdf_parser.chunk_merger import merge_blocks


def _block(content: str, top: float, page: int = 1):
    return {
        "page": page,
        "type": "text",
        "content": content,
        "bbox": [0, top, 100, top + 10],
        "metadata": {"source_file": "sample.pdf", "source_path": "sample.pdf"},
    }


def test_combined_chunking_uses_headings_parent_child_and_recursive_splits():
    chunks = merge_blocks(
        [
            _block("# Overview", 0),
            _block("Alpha. " * 40, 20),
            _block("# Details", 60),
            _block("Beta. " * 40, 80),
        ],
        [],
        max_text_chars=120,
        chunk_strategy="combined",
        parent_text_chars=300,
    )

    assert chunks
    assert all(chunk["metadata"]["chunk_strategy"] == "combined" for chunk in chunks)
    assert chunks[0]["metadata"]["heading"] == "Overview"
    assert "Alpha" in chunks[0]["metadata"]["parent_content"]
    assert chunks[0]["metadata"]["parent_id"] == "parent_0001"
    assert any(chunk["metadata"].get("child_total", 1) > 1 for chunk in chunks)


def test_recursive_chunking_applies_overlap_strategy():
    chunks = merge_blocks(
        [_block("one two three four five six seven eight nine ten " * 10, 0)],
        [],
        max_text_chars=80,
        chunk_overlap=10,
        chunk_strategy="recursive",
    )

    assert len(chunks) > 1
    assert all(chunk["metadata"]["chunk_strategy"] == "recursive" for chunk in chunks)
    assert chunks[0]["chunk_id"] == "p1_001"
