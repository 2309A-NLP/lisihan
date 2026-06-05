from pdf_parser.parser import PDFParser
from src.retriever.hybrid import HybridRetriever


def test_parser_documents_keep_source_file_for_filtered_retrieval():
    parsed = {
        "source_file": "sample.pdf",
        "source_path": "data/sample.pdf",
        "chunks": [
            {
                "type": "text",
                "content": "Acme basic company information and legal representative.",
                "page": 1,
                "chunk_id": "p1_001",
                "metadata": {},
            }
        ],
    }

    docs = PDFParser()._to_documents(parsed)
    assert docs[0].metadata["source_file"] == "sample.pdf"
    assert docs[0].metadata["source_path"] == "data/sample.pdf"

    retriever = HybridRetriever()
    retriever.create_vectorstore(docs)

    assert retriever.search("Acme legal representative", top_k=1, source_file="sample.pdf")
    assert retriever.search("Acme legal representative", top_k=1, source_file="other.pdf") == []
