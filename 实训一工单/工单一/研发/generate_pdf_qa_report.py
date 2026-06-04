# -*- coding: utf-8 -*-
"""Generate deliverables for PDF-RAG answers vs LLM-only answers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.config import Config
from src.rag_engine import RAGEngine
from utils.evaluator import RAGEvaluator


OUTPUT_DIR = Path("outputs")


def _shorten(text: str, limit: int = 320) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _build_markdown(
    payload: Dict,
    rows: List[Dict],
    json_path: Path,
) -> str:
    summary = payload["summary"]
    generated_at = payload["generated_at"]
    lines = [
        "# PDF 问答系统评估与对比分析报告",
        "",
        f"- 生成时间：{generated_at}",
        f"- PDF 目录：`{payload['pdf_dir']}`",
        f"- 检索方式：`{payload['retrieval_backend']}`",
        f"- 索引片段数：{payload['document_count']}",
        f"- 明细 JSON：`{json_path.name}`",
        "",
        "## 一、功能验收结论",
        "",
        "系统已具备基于 PDF 文档内容回答问题的能力：启动时会解析 `data/` 目录中的 PDF，构建 BM25 检索索引，提问后先召回 PDF 片段，再基于片段生成答案。系统也支持不传入 PDF 上下文的纯 LLM 答案，并可将两类答案放在同一问题集下做对比。",
        "",
        "## 二、整体指标",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 问题总数 | {summary['total_queries']} |",
        f"| 成功查询数 | {summary['successful_queries']} |",
        f"| 成功率 | {_percent(summary['success_rate'])} |",
        f"| 严格匹配准确率 | {_percent(summary['accuracy'])} |",
        f"| 平均响应时间 | {summary['avg_response_time']:.3f} 秒 |",
        f"| 平均检索时间 | {summary['avg_retrieval_time'] * 1000:.0f} ms |",
        f"| 平均查询时间 | {summary['avg_query_time'] * 1000:.0f} ms |",
        f"| 平均生成时间 | {summary['avg_generation_time'] * 1000:.0f} ms |",
        f"| 平均总输出时间 | {summary['avg_total_time'] * 1000:.0f} ms |",
        f"| 上下文相关性 | {_percent(summary['context_relevance'])} |",
        f"| 答案忠实度 | {_percent(summary['answer_faithfulness'])} |",
        f"| 答案相关性 | {_percent(summary['answer_relevance'])} |",
        f"| RAG 对比值 | {_percent(summary['rag_vs_llm_improvement'])} |",
        "",
        "## 三、PDF 答案与纯 LLM 答案对比",
        "",
        "| ID | 问题 | PDF/RAG 答案 | 纯 LLM 答案 | 检索片段 | 综合评分 | RAG 对比值 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {id} | {question} | {rag_answer} | {llm_only_answer} | {retrieved_count} | {overall_score:.2f} | {rag_vs_llm:.2f} |".format(
                id=row["id"],
                question=row["question"],
                rag_answer=_shorten(row["rag_answer"]).replace("|", "\\|"),
                llm_only_answer=_shorten(row["llm_only_answer"]).replace("|", "\\|"),
                retrieved_count=row["rag_retrieved_count"],
                overall_score=row["comparison"]["overall_score"],
                rag_vs_llm=row["comparison"].get("rag_vs_llm", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## 四、分析说明",
            "",
            "- PDF/RAG 答案：使用 PDF 检索片段作为上下文生成，优势是可以引用文档中的专有信息、金额、标准、人员等事实。",
            "- 纯 LLM 答案：不接收 PDF 上下文，只依赖模型已有知识；在当前离线兜底或模型无法访问时，会明确返回知识库信息不足。",
            "- 对比指标：上下文相关性衡量召回片段与问题关键词的匹配程度；答案忠实度衡量答案是否可被召回片段支撑；RAG 对比值用于粗略衡量 RAG 相对纯 LLM 的信息增益。",
            "",
            "## 五、可运行性",
            "",
            "本报告由项目代码直接生成，生成过程会完成 PDF 解析、BM25 索引构建、批量问答、纯 LLM 对照回答和评估统计。若在线 LLM 接口不可用，系统会自动使用本地离线兜底逻辑，保证项目仍可跑通。",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_report(use_online_llm: bool = False) -> Dict[str, Path]:
    if not use_online_llm:
        Config.LLM_API_KEY = ""
        Config.OPENAI_API_KEY = ""

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"pdf_qa_comparison_{timestamp}.json"
    md_path = OUTPUT_DIR / f"pdf_qa_comparison_report_{timestamp}.md"

    engine = RAGEngine()
    evaluator = RAGEvaluator()
    init_result = engine.initialize_project_knowledge_base()
    if not init_result.success:
        raise RuntimeError(f"知识库初始化失败：{init_result.message} {init_result.details}")

    batch_results = engine.batch_answer(Config.EVALUATION_QUESTIONS)
    eval_result = evaluator.evaluate_batch(batch_results)

    rows = []
    for item in batch_results:
        comparison = evaluator.evaluate_single(
            question=item["question"],
            rag_answer=item["rag_answer"],
            retrieved_contexts=item.get("retrieved_contexts", []),
            llm_only_answer=item.get("llm_only_answer", ""),
        )
        rows.append(
            {
                **item,
                "comparison": comparison,
                "source_preview": [
                    {
                        "rank": idx + 1,
                        "content": _shorten(ctx, limit=500),
                    }
                    for idx, ctx in enumerate(item.get("retrieved_contexts", []))
                ],
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pdf_dir": Config.PDF_DIR,
        "retrieval_backend": Config.RETRIEVAL_BACKEND,
        "collection_name": Config.COLLECTION_NAME,
        "document_count": init_result.document_count,
        "llm_mode": "online" if use_online_llm else "offline-safe",
        "summary": {
            "total_queries": eval_result.details["total_queries"],
            "successful_queries": eval_result.details["successful_queries"],
            "success_rate": eval_result.success_rate,
            "avg_response_time": eval_result.avg_response_time,
            "accuracy": eval_result.accuracy,
            "accuracy_match_standard": eval_result.details["accuracy"]["match_standard"],
            "accuracy_correct_count": eval_result.details["accuracy"]["correct_count"],
            "avg_retrieval_time": eval_result.avg_retrieval_time,
            "avg_query_time": eval_result.avg_query_time,
            "avg_generation_time": eval_result.avg_generation_time,
            "avg_total_time": eval_result.avg_total_time,
            "context_relevance": eval_result.context_relevance,
            "answer_faithfulness": eval_result.answer_faithfulness,
            "answer_relevance": eval_result.answer_relevance,
            "rag_vs_llm_improvement": eval_result.rag_vs_llm_improvement,
        },
        "results": rows,
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(payload, rows, json_path), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 PDF/RAG 与纯 LLM 答案对比产出物")
    parser.add_argument(
        "--online-llm",
        action="store_true",
        help="使用 .env 中配置的在线 LLM；默认使用离线安全模式以保证可跑通。",
    )
    args = parser.parse_args()

    paths = generate_report(use_online_llm=args.online_llm)
    print(f"Markdown报告: {paths['markdown'].resolve()}")
    print(f"JSON明细: {paths['json'].resolve()}")


if __name__ == "__main__":
    main()
