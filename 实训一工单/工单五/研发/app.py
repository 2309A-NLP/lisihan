# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件属于 PDF 招股说明书智能问答系统，接入工单五多轮对话 session_id，
并保留工单一到工单四的文本检索、图片内容解析和 Redis 缓存能力。
"""

from __future__ import annotations

import os
import sys
import time
import uuid


import streamlit as st

from src.config import Config
from src.memory_manager import LongTermMemoryManager, RedisMemoryManager
from src.rag_engine import RAGEngine
from utils.evaluator import RAGEvaluator
from utils.logger import get_logger


logger = get_logger(__name__)

MODE_LABELS = {
    "hybrid": "混合检索",
    "bm25": "BM25检索",
    "vector": "向量检索",
    "rrf": "RRF融合检索",
    "long_term_memory": "长期记忆",
}

LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "English",
}


st.set_page_config(
    page_title="PDF智能问答系统",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


class PDFQASystem:
    def __init__(self):
        self.evaluator = RAGEvaluator()
        self.memory_manager = RedisMemoryManager()
        self.long_term_memory = LongTermMemoryManager()
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
            "show_short_memory": False,
            "show_long_term_memory": False,
            "retrieval_mode": "hybrid",
            "answer_language": "zh",
            "session_id": f"streamlit-{uuid.uuid4().hex}",
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
            if st.session_state.retrieval_mode not in {"hybrid", "bm25", "vector"}:
                st.session_state.retrieval_mode = "hybrid"
            st.session_state.retrieval_mode = st.selectbox(
                "当前检索模式",
                options=["hybrid", "bm25", "vector"],
                index=["hybrid", "bm25", "vector"].index(st.session_state.retrieval_mode),
                format_func=lambda mode: MODE_LABELS[mode],
            )
            if st.session_state.answer_language not in {"zh", "en"}:
                st.session_state.answer_language = "zh"
            st.session_state.answer_language = st.selectbox(
                "回答语言",
                options=["zh", "en"],
                index=["zh", "en"].index(st.session_state.answer_language),
                format_func=lambda language: LANGUAGE_LABELS[language],
            )

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

            if st.button("重新解析并构建混合检索索引"):
                with st.spinner("正在重建知识库..."):
                    self.reindex()
                st.rerun()

            st.divider()
            if st.button("查看最近 20 条短期对话历史"):
                st.session_state.show_short_memory = True

            if st.session_state.show_short_memory:
                history = self.memory_manager.get_history(limit=20)
                if not self.memory_manager.is_available():
                    st.warning("Redis 短期记忆不可用，请确认 Redis 服务已启动。")
                elif not history:
                    st.info("暂无短期对话历史。")
                else:
                    st.caption(f"最近 {len(history)} 条短期对话")
                    for idx, item in enumerate(history, start=1):
                        role = "用户" if item.get("role") == "user" else "助手"
                        st.markdown(f"**{idx}. {role}**")
                        st.caption(item.get("timestamp", ""))
                        st.write(item.get("content", ""))

            if st.button("查看长期记忆"):
                st.session_state.show_long_term_memory = True
                self.long_term_memory.refresh_status()

            if st.session_state.show_long_term_memory:
                memories = self.long_term_memory.list_memories(limit=Config.LONG_TERM_MEMORY_LIMIT)
                if not self.long_term_memory.is_available():
                    st.warning("Milvus 长期记忆不可用，请确认 Milvus 服务已启动。")
                elif not memories:
                    st.info("暂无长期记忆。")
                else:
                    st.caption(f"长期记忆：{len(memories)} 条优质问答")
                    for idx, item in enumerate(memories, start=1):
                        st.markdown(f"**{idx}. {item.get('question', '')}**")
                        st.caption(f"有帮助次数: {item.get('helpful_count', 0)} | {item.get('timestamp', '')}")
                        st.write(item.get("answer", ""))

            st.divider()
            if st.button("运行工单问题评估"):
                self.run_evaluation()

            if st.session_state.evaluation_results:
                res = st.session_state.evaluation_results
                st.metric("平均响应时间", f"{res.avg_response_time:.3f}s")
                st.metric("成功率", f"{res.success_rate:.1%}")

    def record_feedback(self, helpful: bool):
        last = st.session_state.last_response
        if not last or self.rag_engine is None:
            return
        score = "helpful" if helpful else "needs_improvement"
        st.session_state.feedback.append({**last, "score": score, "timestamp": time.time()})
        if helpful and Config.ENABLE_LONG_TERM_MEMORY_ANSWER:
            saved = self.long_term_memory.save_qa(last["question"], last["answer"])
            if saved:
                st.success("已存入 Milvus 长期记忆。")
            else:
                st.warning("未能写入 Milvus 长期记忆，请检查 Milvus 或向量模型配置。")
        elif helpful:
            st.info("已记录为有帮助；长期记忆自动回答当前已关闭，未写入 Milvus。")
        self.rag_engine.record_feedback(
            question=last["question"],
            answer=last["answer"],
            helpful=helpful,
            question_type=last.get("question_type", "entity"),
        )
        st.success("已记录反馈。")

    def render_chat(self):
        st.title("PDF招股说明书智能问答系统")
        st.caption("当前系统使用 BM25 + 向量检索 + RRF 融合；对话历史仅用于指代消解参考，答案仍由 RAG 检索和多模态解析生成。")

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
        self.memory_manager.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if self.rag_engine is None:
                st.error("知识库未初始化，请确认 data 目录下存在 PDF 并重新解析。")
                return
            try:
                response = self.rag_engine.ask(
                    prompt,
                    retrieval_mode=st.session_state.retrieval_mode,
                    session_id=st.session_state.session_id,
                    answer_language=st.session_state.answer_language,
                )
                answer = response.answer
                st.markdown(answer)

                metadata = {
                    "question_type": response.question_type,
                    "retrieval_mode": response.retrieval_mode,
                    "response_time": response.response_time,
                    "accuracy": response.accuracy,
                    "answer_source": "long_term_memory" if response.memory_hit else "retrieval",
                    "retrieved_count": len(response.retrieved_contexts),
                    "scores": response.scores,
                    "query_analysis": response.query_analysis,
                    "answer_language": st.session_state.answer_language,
                }
                answer_type = "来自长期记忆" if response.memory_hit else "来自混合检索"
                mode_label = MODE_LABELS.get(response.retrieval_mode, response.retrieval_mode)
                st.caption(
                    f"响应时间: {response.response_time:.3f}s | "
                    f"置信度估算: {response.accuracy:.1%} | "
                    f"答案类型: {answer_type} | "
                    f"问题类型: {response.question_type} | "
                    f"检索模式: {mode_label}"
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
                    "accuracy": response.accuracy,
                }
                st.session_state.messages.append({"role": "assistant", "content": answer, "metadata": metadata})
                self.memory_manager.add_message("assistant", answer)

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
                    f"RAG 对比值 {res.rag_vs_llm_improvement:.2%}。"
                )
                st.dataframe(st.session_state.evaluation_comparison, use_container_width=True)

    def run_evaluation(self):
        if self.rag_engine is None:
            st.warning("请先初始化知识库。")
            return
        results = self.rag_engine.batch_answer(
            Config.EVALUATION_QUESTIONS,
            answer_language=st.session_state.answer_language,
        )
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
                    "置信度估算": round(item.get("accuracy", 0.0), 4),
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
