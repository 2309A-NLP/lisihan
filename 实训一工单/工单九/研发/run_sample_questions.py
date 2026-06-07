# -*- coding: utf-8 -*-
"""Run sample_questions.pdf through the optimized Graph RAG engine.

人工智能 NLP-RAG-Graph RAG 优化任务
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from evaluate_with_ragas import DEFAULT_REFERENCE_PDF, OUTPUT_DIR, parse_reference_qa
from src.rag_engine_optimized import RAGEngineOptimized

DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "sample_questions_graph_rag_results.json"


def safe_console(text: str) -> str:
    return str(text).encode("gbk", errors="replace").decode("gbk")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def response_to_result(index: int, question: str, reference_answer: str, response: Any) -> dict[str, Any]:
    query_analysis = to_jsonable(getattr(response, "query_analysis", {}) or {})
    return {
        "id": index,
        "question": question,
        "reference_answer": reference_answer,
        "answer": getattr(response, "answer", ""),
        "question_type": getattr(response, "question_type", ""),
        "accuracy": getattr(response, "accuracy", 0.0),
        "response_time": getattr(response, "response_time", 0.0),
        "retrieval_mode": getattr(response, "retrieval_mode", "hybrid_graph_optimized"),
        "has_context": bool(getattr(response, "has_context", False)),
        "retrieved_count": len(getattr(response, "source_chunks", []) or []),
        "scores": to_jsonable(getattr(response, "scores", []) or []),
        "retrieved_contexts": to_jsonable(getattr(response, "retrieved_contexts", []) or []),
        "source_chunks": to_jsonable(getattr(response, "source_chunks", []) or []),
        "query_analysis": query_analysis,
        "graph_backend": query_analysis.get("graph_backend", ""),
        "status": "success",
        "error": "",
    }


def run_questions(
    *,
    reference_pdf: Path,
    output_path: Path,
    retrieval_mode: str,
    answer_language: str,
    initialize: bool,
) -> dict[str, Any]:
    qa_items = parse_reference_qa(reference_pdf)
    if not qa_items:
        raise RuntimeError(f"No questions parsed from {reference_pdf}")

    engine = RAGEngineOptimized()
    init_started = time.time()
    init_success = True
    init_result = {}
    if initialize:
        if hasattr(engine, "initialize_project_knowledge_base"):
            result = engine.initialize_project_knowledge_base()
            init_success = bool(getattr(result, "success", False))
            init_result = to_jsonable(result)
        else:
            init_success = bool(engine.initialize_from_project())
            init_result = {"success": init_success}
    init_time = time.time() - init_started

    results = []
    failed_count = 0
    for index, item in enumerate(qa_items, start=1):
        question = item["question"]
        reference_answer = item["answer"]
        try:
            response = engine.ask(
                question,
                question_id=index,
                retrieval_mode=retrieval_mode,
                session_id="sample_questions",
                answer_language=answer_language,
            )
            results.append(response_to_result(index, question, reference_answer, response))
            print(safe_console(f"[{index}/{len(qa_items)}] success | {response.response_time:.3f}s | {question[:60]}"))
        except Exception as exc:
            failed_count += 1
            results.append(
                {
                    "id": index,
                    "question": question,
                    "reference_answer": reference_answer,
                    "answer": "",
                    "response_time": 0.0,
                    "retrieval_mode": retrieval_mode,
                    "has_context": False,
                    "retrieved_count": 0,
                    "source_chunks": [],
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(safe_console(f"[{index}/{len(qa_items)}] failed | {question[:60]} | {exc}"))

    response_times = [float(item.get("response_time") or 0.0) for item in results if item.get("response_time")]
    context_count = sum(1 for item in results if item.get("has_context"))
    payload = {
        "summary": {
            "work_order": "人工智能 NLP-RAG-Graph RAG 优化任务",
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reference_pdf": str(reference_pdf),
            "retrieval_mode": retrieval_mode,
            "total": len(results),
            "success_count": len(results) - failed_count,
            "failed_count": failed_count,
            "context_count": context_count,
            "context_rate": context_count / len(results) if results else 0.0,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
            "init_success": init_success,
            "init_time": init_time,
            "init_result": init_result,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run sample_questions.pdf through RAGEngineOptimized.")
    parser.add_argument("--reference-pdf", type=Path, default=DEFAULT_REFERENCE_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--retrieval-mode", default="hybrid_graph_optimized")
    parser.add_argument("--answer-language", default="zh")
    parser.add_argument("--no-init", action="store_true", help="Skip project knowledge-base initialization.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_questions(
        reference_pdf=args.reference_pdf,
        output_path=args.output,
        retrieval_mode=args.retrieval_mode,
        answer_language=args.answer_language,
        initialize=not args.no_init,
    )
    summary = payload["summary"]
    print(f"saved: {args.output}")
    print(f"success: {summary['success_count']}/{summary['total']}")
    print(f"average_response_time: {summary['average_response_time']:.4f}s")
    return payload


if __name__ == "__main__":
    main()
