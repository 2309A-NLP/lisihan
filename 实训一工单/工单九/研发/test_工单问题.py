# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.rag_engine import RAGEngine
from utils.evaluator import RAGEvaluator
from utils.logger import get_logger


logger = get_logger(__name__)


QUESTIONS = [
    {"id": 260, "question": "军用领域收入分别是多少"},
    {"id": 95, "question": "参与制定了哪个技术标准"},
    {"id": 33, "question": "收入占主营业务收入的比重分别是多少"},
    {"id": 34, "question": "上游涉及哪些企业"},
    {"id": 957, "question": "在哪个领域已经成为重要供应商"},
    {"id": 793, "question": "下游主要包括哪些行业"},
    {"id": 795, "question": "哪个工程荣获了国家科技进步一等奖"},
    {"id": 543, "question": "注册资本是多少"},
    {"id": 531, "question": "法定代表人是谁"},
    {"id": 207, "question": "计划使用多少募集资金补充流动资金"},
]


def main() -> None:
    output_path = Path("工单问题测试结果.json")
    engine = RAGEngine()
    evaluator = RAGEvaluator()

    logger.info("workshop test start | count=%s", len(QUESTIONS))
    init_result = engine.initialize_project_knowledge_base()
    if not init_result.success:
        raise RuntimeError(f"知识库初始化失败: {init_result.message} | {init_result.details}")

    results = []
    comparison_rows = []
    success_count = 0

    for item in QUESTIONS:
        start = time.time()
        response = engine.ask(item["question"], question_id=item["id"])
        elapsed = time.time() - start
        llm_answer, llm_time = engine.answer_without_rag(item["question"])
        compare = evaluator.evaluate_single(
            question=item["question"],
            rag_answer=response.answer,
            retrieved_contexts=response.retrieved_contexts,
            llm_only_answer=llm_answer,
        )

        result = {
            "id": item["id"],
            "question": item["question"],
            "answer": response.answer,
            "question_type": response.question_type,
            "retrieval_mode": response.retrieval_mode,
            "response_time": response.response_time,
            "elapsed_time": elapsed,
            "has_context": response.has_context,
            "retrieved_count": len(response.retrieved_contexts),
            "query_analysis": response.query_analysis,
            "llm_only_answer": llm_answer,
            "llm_only_response_time": llm_time,
            "comparison": compare,
        }
        results.append(result)
        comparison_rows.append(
            {
                "question": item["question"],
                "rag_answer": response.answer,
                "llm_only_answer": llm_answer,
                "context_relevance": compare["context_relevance"],
                "answer_faithfulness": compare["answer_faithfulness"],
                "answer_completeness": compare["answer_completeness"],
                "overall_score": compare["overall_score"],
                "rag_vs_llm": compare.get("rag_vs_llm", 0.0),
            }
        )

        if response.answer and response.answer.strip():
            success_count += 1

        print(f"[{item['id']}] {item['question']}")
        print(f"类型: {response.question_type}")
        print(f"检索模式: {response.retrieval_mode}")
        print(f"答案: {response.answer}")
        print(f"纯LLM答案: {llm_answer}")
        print(
            "对比: "
            f"相关性={compare['context_relevance']:.2f}, "
            f"忠实度={compare['answer_faithfulness']:.2f}, "
            f"综合={compare['overall_score']:.2f}, "
            f"RAG对比值={compare.get('rag_vs_llm', 0.0):.2f}"
        )
        print(f"响应时间: {response.response_time:.3f}s")
        print("-")

    payload = {
        "total": len(QUESTIONS),
        "success_count": success_count,
        "results": results,
        "comparison": comparison_rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"成功数量: {success_count}/{len(QUESTIONS)}")
    print(f"结果已保存到: {output_path.resolve()}")
    logger.info("workshop test done | success=%s | total=%s | output=%s", success_count, len(QUESTIONS), output_path)


if __name__ == "__main__":
    main()

