# -*- coding: utf-8 -*-
"""
主程序入口
"""

import os
import sys
import subprocess
from pathlib import Path


def _project_python() -> Path:
    project_root = Path(__file__).resolve().parent
    if os.name == "nt":
        return project_root / ".venv" / "Scripts" / "python.exe"
    return project_root / ".venv" / "bin" / "python"


def ensure_project_venv():
    """避免用其他项目的虚拟环境启动当前项目。"""
    target_python = _project_python()
    if not target_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    try:
        already_using_project_venv = current_python.samefile(target_python)
    except OSError:
        already_using_project_venv = str(current_python).lower() == str(target_python.resolve()).lower()

    if already_using_project_venv:
        return

    print(f"检测到当前 Python 环境不是本项目 .venv，切换到: {target_python}")
    subprocess.run([str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]], check=False)
    sys.exit(0)


def check_dependencies():
    required_packages = [
        "streamlit",
        "rank_bm25",
        "fitz",
        "pdfplumber",
        "redis",
        "pymilvus",
        "sentence_transformers",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"缺少以下依赖包: {missing}")
        print("请先在项目 .venv 中安装对应依赖。")
        return False
    return True


def run_cli_demo():
    from src.rag_engine import RAGEngine
    from utils.evaluator import RAGEvaluator
    from src.config import Config
    from utils.logger import get_logger

    logger = get_logger(__name__)

    print("=" * 60)
    print("PDF RAG问答系统 - 命令行演示")
    print(f"项目编号: {Config.TASK_ID}")
    print("=" * 60)
    print("说明: 当前项目使用 BM25 + 向量检索 + RRF 融合运行。")
    logger.info("cli demo start")

    rag = RAGEngine()
    evaluator = RAGEvaluator()
    print("\n正在初始化系统...")

    init_result = rag.initialize_project_knowledge_base()
    print(init_result.message)
    if init_result.details:
        print(init_result.details)
    if not init_result.success:
        print("请将PDF文件放入 ./data 目录，或检查知识库配置。")
        logger.warning("cli init failed | status=%s | details=%s", init_result.status, init_result.details)
        return

    print(f"\n开始评估 {len(Config.EVALUATION_QUESTIONS)} 个问题...\n")

    results = []
    comparison_rows = []
    for q in Config.EVALUATION_QUESTIONS:
        print(f"问题 {q['id']}: {q['question']}")
        response = rag.answer(q["question"])
        print(f"  RAG回答: {response.answer[:150]}...")
        print(f"  响应时间: {response.response_time:.3f}秒")
        llm_answer, llm_time = rag.answer_without_rag(q["question"])
        print(f"  纯LLM回答: {llm_answer[:150]}...")
        print(f"  纯LLM时间: {llm_time:.3f}秒")
        single = evaluator.evaluate_single(
            question=q["question"],
            rag_answer=response.answer,
            retrieved_contexts=response.retrieved_contexts,
            llm_only_answer=llm_answer,
        )
        print(
            "  对比分析: "
            f"上下文相关性={single['context_relevance']:.2f}, "
            f"忠实度={single['answer_faithfulness']:.2f}, "
            f"综合评分={single['overall_score']:.2f}, "
            f"RAG对比值={single.get('rag_vs_llm', 0.0):.2f}"
        )
        print("-" * 40)
        results.append({
            "question": q["question"],
            "rag_answer": response.answer,
            "llm_only_answer": llm_answer,
            "retrieved_contexts": response.retrieved_contexts,
            "has_context": response.has_context,
            "response_time": response.response_time
        })
        comparison_rows.append({
            "question": q["question"],
            "rag_answer": response.answer,
            "llm_only_answer": llm_answer,
            "context_relevance": single["context_relevance"],
            "answer_faithfulness": single["answer_faithfulness"],
            "answer_completeness": single["answer_completeness"],
            "overall_score": single["overall_score"],
            "rag_vs_llm": single.get("rag_vs_llm", 0.0),
        })

    eval_result = evaluator.evaluate_batch(results)
    evaluator.print_evaluation_report(eval_result)
    print("\n对比明细:")
    for row in comparison_rows[:3]:
        print(f"- {row['question']}")
        print(f"  RAG: {row['rag_answer'][:120]}")
        print(f"  LLM: {row['llm_only_answer'][:120]}")
        print(f"  评分: {row['overall_score']:.2f} / RAG对比值: {row['rag_vs_llm']:.2f}")
    logger.info("cli demo finished")


def run_web_app():
    print("启动Web应用...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8502",
        "--server.address", "localhost"
    ])


def main():
    ensure_project_venv()

    if not check_dependencies():
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="PDF RAG问答系统")
    parser.add_argument("--mode", choices=["cli", "web"], default="web",
                        help="运行模式: cli(命令行) 或 web(网页)")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli_demo()
    else:
        run_web_app()


if __name__ == "__main__":
    main()

