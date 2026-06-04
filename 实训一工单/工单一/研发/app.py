# -*- coding: utf-8 -*-
"""Streamlit app for the PDF RAG Q&A system."""

from __future__ import annotations

import time

import streamlit as st

from src.config import Config
from src.rag_engine import RAGEngine
from utils.evaluator import RAGEvaluator
from utils.logger import get_logger


logger = get_logger(__name__)


st.set_page_config(
    page_title="PDF智能问答系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


class PDFQASystem:
    def __init__(self):
        self.evaluator = RAGEvaluator()
        self.init_session_state()
        self.rag_engine = self.get_or_create_engine()

    def init_session_state(self):
        defaults = {
            "messages": [],
            "feedback": [],
            "rag_engine": None,
            "init_status": None,
            "system_initialized": False,
            "evaluation_results": None,
            "evaluation_comparison": [],
            "last_response": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_or_create_engine(self):
        if st.session_state.rag_engine is not None:
            return st.session_state.rag_engine
        engine = RAGEngine()
        result = engine.initialize_project_knowledge_base()
        st.session_state.init_status = result
        st.session_state.system_initialized = result.success
        if result.success:
            st.session_state.rag_engine = engine
            return engine
        return None

    def reindex(self):
        engine = RAGEngine()
        result = engine.initialize_project_knowledge_base()
        st.session_state.init_status = result
        st.session_state.system_initialized = result.success
        st.session_state.rag_engine = engine if result.success else None
        self.rag_engine = st.session_state.rag_engine

    def render_sidebar(self):
        with st.sidebar:
            st.title("配置面板")
            st.write(f"检索方式: `{Config.RETRIEVAL_BACKEND}`")
            st.write(f"索引名称: `{Config.COLLECTION_NAME}`")
            st.write(f"召回条数: `{Config.TOP_K_RETRIEVAL}`")

            result = st.session_state.init_status
            if result and result.success:
                st.success(result.message)
                if result.document_count:
                    st.caption(f"文档片段数: {result.document_count}")
            elif result:
                st.error(result.message)
                st.code(result.details)
            else:
                st.warning("知识库尚未初始化")

            if st.button("重新解析并构建 BM25 索引"):
                with st.spinner("正在重建知识库..."):
                    self.reindex()
                st.rerun()

            st.divider()
            if st.button("运行工单问题评估"):
                self.run_evaluation()

            if st.session_state.evaluation_results:
                res = st.session_state.evaluation_results
                st.metric("准确率", f"{res.accuracy:.1%}")
                st.metric("平均总输出时间", f"{res.avg_total_time * 1000:.0f}ms")

    def record_feedback(self, helpful: bool):
        last = st.session_state.last_response
        if not last or self.rag_engine is None:
            return
        score = "helpful" if helpful else "needs_improvement"
        st.session_state.feedback.append({**last, "score": score, "timestamp": time.time()})
        self.rag_engine.record_feedback(
            question=last["question"],
            answer=last["answer"],
            helpful=helpful,
            question_type=last.get("question_type", "entity"),
        )
        st.success("已记录反馈。")

    def render_chat(self):
        st.title("PDF招股说明书智能问答系统")
        st.caption("当前系统仅使用 BM25 关键词检索，不保存短期记忆或长期记忆。")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    with st.expander("调试信息"):
                        st.json(msg["metadata"])

        prompt = st.chat_input("请输入你的问题")
        if not prompt:
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if self.rag_engine is None:
                st.error("知识库未初始化，请确认 data 目录下存在 PDF 并重新解析。")
                return
            try:
                response = self.rag_engine.ask(prompt)
                placeholder = st.empty()
                content = ""
                for i in range(0, len(response.answer), 18):
                    content += response.answer[i : i + 18]
                    placeholder.markdown(content)
                    time.sleep(0.01)
                answer = content or response.answer

                metadata = {
                    "question_type": response.question_type,
                    "retrieval_mode": response.retrieval_mode,
                    "response_time": response.response_time,
                    "retrieval_time": response.retrieval_time,
                    "query_time": response.query_time,
                    "generation_time": response.generation_time,
                    "total_time": response.total_time,
                    "retrieved_count": len(response.retrieved_contexts),
                    "scores": response.scores,
                    "query_analysis": response.query_analysis,
                }
                st.caption(
                    f"检索: {response.retrieval_time * 1000:.0f}ms | "
                    f"查询: {response.query_time * 1000:.0f}ms | "
                    f"生成: {response.generation_time * 1000:.0f}ms | "
                    f"总输出: {response.total_time * 1000:.0f}ms | "
                    f"类型: {response.question_type} | 检索: BM25"
                )

                if response.retrieved_contexts:
                    with st.expander("查看检索片段"):
                        for idx, ctx in enumerate(response.retrieved_contexts, start=1):
                            score = response.scores[idx - 1] if idx - 1 < len(response.scores) else 0
                            st.markdown(f"**片段 {idx} | 分数 {score:.4f}**")
                            st.info(ctx[:800] + ("..." if len(ctx) > 800 else ""))

                st.session_state.last_response = {
                    "question": prompt,
                    "answer": answer,
                    "question_type": response.question_type,
                }
                st.session_state.messages.append({"role": "assistant", "content": answer, "metadata": metadata})

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("有帮助", key=f"helpful_{len(st.session_state.messages)}"):
                        self.record_feedback(True)
                with col2:
                    if st.button("需改进", key=f"bad_{len(st.session_state.messages)}"):
                        self.record_feedback(False)
            except Exception as exc:
                logger.exception("chat failed")
                st.error("回答生成失败，详情已写入日志。")
                st.code(str(exc))

        if st.session_state.evaluation_results and st.session_state.evaluation_comparison:
            with st.expander("查看 PDF 答案与纯 LLM 答案对比分析", expanded=False):
                res = st.session_state.evaluation_results
                st.write(
                    f"整体结论：上下文相关性 {res.context_relevance:.2%}，"
                    f"答案忠实度 {res.answer_faithfulness:.2%}，"
                    f"答案相关性 {res.answer_relevance:.2%}，"
                    f"严格匹配准确率 {res.accuracy:.2%}，"
                    f"RAG 对比值 {res.rag_vs_llm_improvement:.2%}。"
                )
                st.dataframe(st.session_state.evaluation_comparison, use_container_width=True)

    def run_evaluation(self):
        if self.rag_engine is None:
            st.warning("请先初始化知识库。")
            return
        results = self.rag_engine.batch_answer(Config.EVALUATION_QUESTIONS)
        eval_result = self.evaluator.evaluate_batch(results)
        comparison_rows = []
        for item in results:
            single = self.evaluator.evaluate_single(
                question=item["question"],
                rag_answer=item["rag_answer"],
                retrieved_contexts=item.get("retrieved_contexts", []),
                llm_only_answer=item.get("llm_only_answer", ""),
            )
            comparison_rows.append(
                {
                    "问题": item["question"],
                    "PDF答案": item["rag_answer"],
                    "纯LLM答案": item.get("llm_only_answer", ""),
                    "上下文相关性": round(single["context_relevance"], 4),
                    "答案忠实度": round(single["answer_faithfulness"], 4),
                    "答案完整度": round(single["answer_completeness"], 4),
                    "综合评分": round(single["overall_score"], 4),
                    "RAG对比值": round(single.get("rag_vs_llm", 0.0), 4),
                    "严格匹配": next(
                        (
                            row["correct"]
                            for row in eval_result.details["accuracy"]["matches"]
                            if row["id"] == item.get("id")
                        ),
                        False,
                    ),
                    "检索时间(ms)": round(item.get("retrieval_time", 0) * 1000),
                    "查询时间(ms)": round(item.get("query_time", 0) * 1000),
                    "生成时间(ms)": round(item.get("generation_time", 0) * 1000),
                    "总输出时间(ms)": round(item.get("total_time", 0) * 1000),
                    "检索片段数": item.get("rag_retrieved_count", 0),
                }
            )
        st.session_state.evaluation_results = eval_result
        st.session_state.evaluation_comparison = comparison_rows
        st.success("评估完成。")

    def run(self):
        self.render_sidebar()
        self.render_chat()


def main():
    PDFQASystem().run()


if __name__ == "__main__":
    main()
