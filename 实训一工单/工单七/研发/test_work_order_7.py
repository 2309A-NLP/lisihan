# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-功能测试及评估。

独立测试脚本：用于工单七的 10 个问题批量测试。
不修改 src/ 目录中的任何原有代码。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.rag_engine import RAGEngine

RESULTS_PATH = Path("work_order_7_results.json")
REPORT_PATH = Path("work_order_7_report.md")
RETRIEVAL_MODE = "hybrid"
PDF_DIR = Path("data")
EXPECTED_PDF_COUNT = 9

# 工单七：金融年报关键词到文件名的映射
FINANCIAL_REPORT_MAPPING = {
    "平安银行": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
    "中国平安": "2020-02-21__中国平安保险集团股份有限公司__601318__中国平安__2019年__年度报告.pdf",
    "招商银行": "2020-03-21__招商银行股份有限公司__600036__招商银行__2019年__年度报告.pdf",
    "邮储银行": "2020-03-26__中国邮政储蓄银行股份有限公司__601658__邮储银行__2019年__年度报告.pdf",
    "中信证券": "2021-03-19__中信证券股份有限公司__600030__中信证券__2020年__年度报告.pdf",
    "中国人寿": "2021-03-26__中国人寿保险股份有限公司__601628__中国人寿__2020年__年度报告.pdf",
    "中国太保": "2022-03-28__中国太平洋保险集团股份有限公司__601601__中国太保__2021年__年度报告.pdf",
    "招商证券": "2022-03-28__招商证券股份有限公司__600999__招商证券__2021年__年度报告.pdf",
    "国泰君安": "2022-03-31__国泰君安证券股份有限公司__601211__国泰君安__2021年__年度报告.pdf",
}


def get_source_file_for_question(question: str) -> Optional[str]:
    """根据问题内容，返回应该检索的PDF文件名"""
    for keyword, filename in FINANCIAL_REPORT_MAPPING.items():
        if keyword in question:
            return filename
    return None


class PatchedRAGEngine(RAGEngine):
    """继承原引擎，重写 select_pdf 方法"""

    def select_pdf(self, question: str) -> str:
        """重写：根据问题返回正确的PDF文件名"""
        # 工单七：金融年报优先匹配
        for keyword, filename in FINANCIAL_REPORT_MAPPING.items():
            if keyword in question:
                return filename

        # 工单一到工单六：招股说明书
        if "武汉兴图" in question or "兴图新科" in question:
            return "招股说明书1.pdf"
        if "武汉力源" in question or "力源信息" in question:
            return "招股说明书2.pdf"

        # 默认返回原逻辑（不限制）
        return None


# 10个测试问题
test_questions = [
    "平安银行的董事长致辞的时候说了几个方面，分别是什么",
    "中国平安保险（集团）股份有限公司的法定代表人是谁",
    "招商银行股份有限公司第十届董事会第四十七次会议审议通过什么议案",
    "2019年，中国邮政储蓄银行股份有限公司坚持什么原则",
    "分析这些银⾏和保险公司在⾯对经济周期波动时的共同策略与差异化策略，重点关注其在⻛险管理、资本结构优化和新兴领域（如绿⾊⾦融或科技⾦融）投资上的表现。如何评估这些策略在未来经济下⾏周期中的可持续性？",
    "中信证券股份有限公司有多少证券分公司，分别在哪里",
    "中国人寿保险股份有限公司净利润下降的主要原因是什么？",
    "招商证券股份有限公司在2021年信用减值损失是多少？同比增长多少？",
    "中国太平洋保险集团股份有限公司的“哑铃型”资产配置策略是什么？",
    "国泰君安证券股份有限公司调整了哪些组织架构？",
]


def run_tests(engine: PatchedRAGEngine) -> list[dict]:
    results = []

    for i, q in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] 正在测试: {q}")
        start = time.time()

        # 获取当前问题应该使用的PDF
        source_file = get_source_file_for_question(q)
        print(f"    -> 目标PDF: {source_file if source_file else '不限（使用默认路由）'}")

        try:
            resp = engine.ask(q, retrieval_mode=RETRIEVAL_MODE)
            elapsed = time.time() - start
            timing = resp.query_analysis.get("timing", {}) if resp.query_analysis else {}
            retrieval_time = float(timing.get("retrieval_time", 0.0) or 0.0)
            generation_time = float(timing.get("generation_time", 0.0) or 0.0)
            other_time = float(timing.get("other_time", max(0.0, elapsed - retrieval_time - generation_time)) or 0.0)
            result = {
                "id": i,
                "question": q,
                "answer": resp.answer,
                "accuracy": resp.accuracy,
                "response_time": elapsed,
                "engine_response_time": resp.response_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "other_time": other_time,
                "retrieval_mode": resp.retrieval_mode,
                "has_context": resp.has_context,
                "target_pdf": source_file,
                "status": "success",
                "error": "",
            }
            print(f"    -> 答案: {resp.answer[:100]}..." if len(resp.answer) > 100 else f"    -> 答案: {resp.answer}")
            print(f"    -> 准确率: {resp.accuracy * 100:.0f}% | 耗时: {elapsed:.2f}s")
            print(
                "    -> Timing: "
                f"retrieval={retrieval_time:.3f}s | "
                f"generation={generation_time:.3f}s | "
                f"other={other_time:.3f}s | "
                f"engine_total={resp.response_time:.3f}s | "
                f"script_total={elapsed:.3f}s"
            )
        except Exception as exc:
            elapsed = time.time() - start
            result = {
                "id": i,
                "question": q,
                "answer": "",
                "accuracy": 0.0,
                "response_time": elapsed,
                "engine_response_time": 0.0,
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "other_time": elapsed,
                "retrieval_mode": RETRIEVAL_MODE,
                "has_context": False,
                "target_pdf": source_file,
                "status": "failed",
                "error": str(exc),
            }
            print(f"    -> 失败: {exc}")

        results.append(result)

    return results


def print_pdf_check() -> None:
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDF目录: {PDF_DIR.resolve()}")
    print(f"发现PDF数量: {len(pdf_files)}")
    if len(pdf_files) != EXPECTED_PDF_COUNT:
        print(f"提示: 当前发现 {len(pdf_files)} 个PDF，预期工单七测试PDF数量为 {EXPECTED_PDF_COUNT} 个。")
        print("initialize_project_knowledge_base() 会按现有配置解析 PDF_DIR 下的全部PDF。")

    for pdf_file in pdf_files:
        print(f"- {pdf_file.name}")


def build_summary(results: list[dict]) -> dict:
    total = len(results)
    success_count = sum(1 for item in results if item["status"] == "success")
    context_count = sum(1 for item in results if item["has_context"])
    avg_accuracy = sum(item["accuracy"] for item in results) / total if total else 0.0
    avg_response_time = sum(item["response_time"] for item in results) / total if total else 0.0
    avg_retrieval_time = sum(item.get("retrieval_time", 0.0) for item in results) / total if total else 0.0
    avg_generation_time = sum(item.get("generation_time", 0.0) for item in results) / total if total else 0.0
    avg_other_time = sum(item.get("other_time", 0.0) for item in results) / total if total else 0.0

    return {
        "work_order": "人工智能 NLP-RAG-功能测试及评估",
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval_mode": RETRIEVAL_MODE,
        "total": total,
        "success_count": success_count,
        "failed_count": total - success_count,
        "context_count": context_count,
        "context_rate": context_count / total if total else 0.0,
        "average_accuracy": avg_accuracy,
        "average_response_time": avg_response_time,
        "average_retrieval_time": avg_retrieval_time,
        "average_generation_time": avg_generation_time,
        "average_other_time": avg_other_time,
    }


def save_results(summary: dict, results: list[dict]) -> None:
    payload = {
        "summary": summary,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_report(summary: dict, results: list[dict]) -> None:
    lines = [
        "# 工单七 RAG 功能测试及评估报告",
        "",
        "## 汇总",
        "",
        f"- 测试时间: {summary['test_time']}",
        f"- 检索模式: {summary['retrieval_mode']}",
        f"- 测试问题数: {summary['total']}",
        f"- 成功数: {summary['success_count']}",
        f"- 失败数: {summary['failed_count']}",
        f"- 有上下文数量: {summary['context_count']}",
        f"- 上下文命中率: {summary['context_rate']:.2%}",
        f"- 平均准确率: {summary['average_accuracy']:.2%}",
        f"- 平均响应时间: {summary['average_response_time']:.2f}s",
        "",
        "## 明细",
        "",
        "| ID | 问题 | 准确率 | 响应时间 | 检索模式 | 目标PDF | 状态 |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]

    for item in results:
        question = str(item["question"]).replace("|", "\\|")
        target_pdf = item.get("target_pdf", "不限")
        target_pdf_display = target_pdf if target_pdf else "不限"
        lines.append(
            f"| {item['id']} | {question} | {item['accuracy']:.2%} | "
            f"{item['response_time']:.2f}s | {item['retrieval_mode']} | "
            f"{target_pdf_display} | {item['status']} |"
        )

    lines.extend(
        [
            "",
            "## 答案详情",
            "",
        ]
    )

    for item in results:
        lines.extend(
            [
                f"### {item['id']}. {item['question']}",
                "",
                f"- 准确率: {item['accuracy']:.2%}",
                f"- 响应时间: {item['response_time']:.2f}s",
                f"- 检索模式: {item['retrieval_mode']}",
                f"- 目标PDF: {item.get('target_pdf', '不限')}",
                "",
                "答案:",
                "",
                item["answer"] or f"测试失败: {item['error']}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("工单七测试汇总")
    print("=" * 60)
    print(f"测试问题数: {summary['total']}")
    print(f"成功数: {summary['success_count']}")
    print(f"失败数: {summary['failed_count']}")
    print(f"有上下文数量: {summary['context_count']}")
    print(f"上下文命中率: {summary['context_rate']:.2%}")
    print(f"平均准确率: {summary['average_accuracy']:.2%}")
    print(f"平均响应时间: {summary['average_response_time']:.2f}s")
    print(f"JSON结果: {RESULTS_PATH.resolve()}")
    print(f"评估报告: {REPORT_PATH.resolve()}")


def save_report(summary: dict, results: list[dict]) -> None:
    lines = [
        "# Work Order 7 RAG Timing Report",
        "",
        "## Summary",
        "",
        f"- Test time: {summary['test_time']}",
        f"- Retrieval mode: {summary['retrieval_mode']}",
        f"- Total questions: {summary['total']}",
        f"- Success count: {summary['success_count']}",
        f"- Failed count: {summary['failed_count']}",
        f"- Context count: {summary['context_count']}",
        f"- Context rate: {summary['context_rate']:.2%}",
        f"- Average accuracy: {summary['average_accuracy']:.2%}",
        f"- Average response time: {summary['average_response_time']:.3f}s",
        f"- Average retrieval time: {summary['average_retrieval_time']:.3f}s",
        f"- Average generation time: {summary['average_generation_time']:.3f}s",
        f"- Average other time: {summary['average_other_time']:.3f}s",
        "",
        "## Details",
        "",
        "| ID | Question | Accuracy | Script Total | Engine Total | Retrieval | Generation | Other | Mode | Target PDF | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]

    for item in results:
        question = str(item["question"]).replace("|", "\\|")
        target_pdf = item.get("target_pdf") or "unrestricted"
        lines.append(
            f"| {item['id']} | {question} | {item['accuracy']:.2%} | "
            f"{item['response_time']:.3f}s | "
            f"{item.get('engine_response_time', 0.0):.3f}s | "
            f"{item.get('retrieval_time', 0.0):.3f}s | "
            f"{item.get('generation_time', 0.0):.3f}s | "
            f"{item.get('other_time', 0.0):.3f}s | "
            f"{item['retrieval_mode']} | {target_pdf} | {item['status']} |"
        )

    lines.extend(["", "## Answers", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['id']}. {item['question']}",
                "",
                f"- Accuracy: {item['accuracy']:.2%}",
                f"- Script total: {item['response_time']:.3f}s",
                f"- Engine total: {item.get('engine_response_time', 0.0):.3f}s",
                f"- Retrieval time: {item.get('retrieval_time', 0.0):.3f}s",
                f"- Generation time: {item.get('generation_time', 0.0):.3f}s",
                f"- Other time: {item.get('other_time', 0.0):.3f}s",
                f"- Retrieval mode: {item['retrieval_mode']}",
                f"- Target PDF: {item.get('target_pdf') or 'unrestricted'}",
                "",
                "Answer:",
                "",
                item["answer"] or f"Test failed: {item['error']}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("Work Order 7 Timing Summary")
    print("=" * 60)
    print(f"Total questions: {summary['total']}")
    print(f"Success count: {summary['success_count']}")
    print(f"Failed count: {summary['failed_count']}")
    print(f"Context count: {summary['context_count']}")
    print(f"Context rate: {summary['context_rate']:.2%}")
    print(f"Average accuracy: {summary['average_accuracy']:.2%}")
    print(f"Average response time: {summary['average_response_time']:.3f}s")
    print(f"Average retrieval time: {summary['average_retrieval_time']:.3f}s")
    print(f"Average generation time: {summary['average_generation_time']:.3f}s")
    print(f"Average other time: {summary['average_other_time']:.3f}s")
    print(f"JSON results: {RESULTS_PATH.resolve()}")
    print(f"Report: {REPORT_PATH.resolve()}")


def main() -> None:
    print_pdf_check()

    # 使用 PatchedRAGEngine 替代原版
    engine = PatchedRAGEngine()
    init_result = engine.initialize_project_knowledge_base()
    if not init_result.success:
        raise RuntimeError(f"知识库初始化失败: {init_result.message} | {init_result.details}")

    print(f"知识库初始化成功: {init_result.message}")
    if init_result.document_count:
        print(f"文档片段数: {init_result.document_count}")

    results = run_tests(engine)
    summary = build_summary(results)

    save_results(summary, results)
    save_report(summary, results)
    print_summary(summary)


if __name__ == "__main__":
    main()
