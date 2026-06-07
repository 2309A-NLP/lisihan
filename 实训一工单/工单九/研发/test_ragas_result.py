# -*- coding: utf-8 -*-
"""Smoke test RAGAS on the aligned sample_questions Graph RAG results.

人工智能 NLP-RAG-Graph RAG 优化任务
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from evaluate_with_ragas import (
    DEFAULT_REFERENCE_PDF,
    OUTPUT_DIR,
    evaluate_records,
    load_work_order_results,
    parse_reference_qa,
    save_json,
)

DEFAULT_RESULTS_PATH = OUTPUT_DIR / "sample_questions_graph_rag_results.json"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "ragas_test_result.json"


def ragas_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "openai_model": os.getenv("OPENAI_MODEL", ""),
    }
    try:
        import ragas

        info["ragas_installed"] = True
        info["ragas_version"] = getattr(ragas, "__version__", "unknown")
    except Exception as exc:
        info["ragas_installed"] = False
        info["ragas_error"] = str(exc)
    return info


def run_ragas_test(results_path: Path, reference_pdf: Path, output_path: Path) -> dict[str, Any]:
    started = time.time()
    references = parse_reference_qa(reference_pdf)
    payload = load_work_order_results(results_path)

    # prefer_ragas=True means: try real RAGAS first, then fall back to local metrics
    # if no evaluator LLM/API configuration is available.
    evaluation = evaluate_records(
        payload,
        references,
        prefer_ragas=True,
        allow_answer_fallback=False,
    )
    evaluation["summary"].update(
        {
            "results_path": str(results_path),
            "reference_pdf": str(reference_pdf),
            "reference_qa_count": len(references),
            "environment": ragas_environment(),
            "elapsed_seconds": time.time() - started,
        }
    )
    save_json(evaluation, output_path)
    return evaluation


def print_result(result: dict[str, Any], output_path: Path) -> None:
    summary = result["summary"]
    print("RAGAS test result")
    print(f"evaluator: {summary.get('evaluator')}")
    print(f"ragas_available: {summary.get('ragas_available')}")
    print(f"context_precision: {float(summary.get('context_precision') or 0.0):.4f}")
    print(f"context_recall: {float(summary.get('context_recall') or 0.0):.4f}")
    print(f"average_response_time: {float(summary.get('average_response_time') or 0.0):.4f}s")
    print(f"matched_reference_count: {summary.get('matched_reference_count')}")
    print(f"fallback_reference_count: {summary.get('fallback_reference_count')}")
    ragas_attempt = summary.get("ragas_attempt") or {}
    if ragas_attempt.get("error"):
        print(f"ragas_error: {ragas_attempt['error']}")
    print(f"saved: {output_path}")


def main(argv: Iterable[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Test RAGAS on sample_questions_graph_rag_results.json.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--reference-pdf", type=Path, default=DEFAULT_REFERENCE_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = run_ragas_test(args.results, args.reference_pdf, args.output)
    print_result(result, args.output)
    return result


if __name__ == "__main__":
    main()
