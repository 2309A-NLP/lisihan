# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-混合检索任务
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件属于 PDF 招股说明书智能问答系统，接入工单五多轮对话 session_id，
并保留工单一到工单四的文本检索、图片内容解析和 Redis 缓存能力。
"""

from __future__ import annotations

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
    "hybrid_graph": "Graph RAG + 混合检索",
    "graph_only": "纯图谱检索",
    "graph_rag": "Graph RAG",
    "graph": "Graph RAG",
    "hybrid": "混合检索",
    "bm25": "BM25检索",
    "vector": "向量检索",
    "rrf": "RRF融合检索",
    "long_term_memory": "长期记忆",
}

RERANKER_LABELS = {
    "llm": "LLM重排",
    "tfidf": "TF-IDF重排",
    "adaptive": "自适应重排",
}

FUSION_LABELS = {
    "rrf": "RRF倒数排名融合",
    "weighted_average": "加权平均融合",
}

MATCH_TYPE_LABELS = {
    "phrase": "短语匹配",
    "boolean": "布尔查询",
    "fuzzy": "模糊匹配",
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
            "retrieval_mode": Config.RETRIEVAL_CONFIG["mode"],
            "retrieval_reranker": Config.RETRIEVAL_CONFIG["vector"]["reranker"],
            "retrieval_match_type": Config.RETRIEVAL_CONFIG["bm25"]["match_type"],
            "retrieval_fusion": Config.RETRIEVAL_CONFIG["hybrid"]["fusion"],
            "auto_hybrid_weights": True,
            "vector_weight": Config.RETRIEVAL_CONFIG["hybrid"]["vector_weight"],
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

    def initialize_index(self):
        if st.session_state.rag_engine is None:
            st.session_state.rag_engine = RAGEngine()
        result = st.session_state.rag_engine.initialize_project_knowledge_base()
        st.session_state.init_status = result
        st.session_state.system_initialized = result.success
        self.rag_engine = st.session_state.rag_engine if result.success else None
        if not result.success:
            st.session_state.rag_engine = None

    def render_index_panel(self):
        st.subheader("索引管理")
        st.write(f"检索后端: `{Config.RETRIEVAL_BACKEND}`")
        st.write(f"索引名称: `{Config.COLLECTION_NAME}`")
        st.write(f"召回条数: `{Config.TOP_K_RETRIEVAL}`")
        if self.rag_engine is not None and hasattr(self.rag_engine, "graph_store"):
            graph_stats = self.rag_engine.graph_store.stats()
            st.write(f"图谱后端: `{graph_stats.backend}`")
            st.caption(
                f"图谱节点：{graph_stats.node_count} | "
                f"关系：{graph_stats.relation_count} | "
                f"Neo4j：{'已连接' if graph_stats.neo4j_available else '内存模式'}"
            )

        result = st.session_state.init_status
        if result and result.success:
            st.success(result.message)
            st.caption(f"索引状态：已初始化 | 文档片段：{result.document_count or 0}")
            if result.details:
                st.caption(result.details)
        elif result:
            st.error(result.message)
            if result.details:
                st.code(result.details)
        else:
            st.warning("知识库尚未初始化")

        col_init, col_rebuild = st.columns(2)
        with col_init:
            if st.button("初始化索引", use_container_width=True):
                with st.spinner("正在初始化索引..."):
                    self.initialize_index()
                st.rerun()
        with col_rebuild:
            if st.button("重新构建索引", use_container_width=True):
                with st.spinner("正在重新解析 PDF 并构建索引..."):
                    self.reindex()
                st.rerun()

    def render_sidebar(self):
        with st.sidebar:
            st.title("配置面板")
            self.render_index_panel()
            st.divider()
            st.subheader("检索策略")
            retrieval_modes = ["hybrid_graph", "graph_only", "graph_rag", "hybrid", "bm25", "vector"]
            if st.session_state.retrieval_mode not in set(retrieval_modes):
                st.session_state.retrieval_mode = "hybrid_graph"
            st.session_state.retrieval_mode = st.selectbox(
                "当前检索模式",
                options=retrieval_modes,
                index=retrieval_modes.index(st.session_state.retrieval_mode),
                format_func=lambda mode: MODE_LABELS[mode],
            )
            st.session_state.retrieval_match_type = st.selectbox(
                "BM25匹配类型",
                options=["phrase", "boolean", "fuzzy"],
                index=["phrase", "boolean", "fuzzy"].index(st.session_state.retrieval_match_type),
                format_func=lambda item: MATCH_TYPE_LABELS[item],
            )
            st.session_state.retrieval_reranker = st.selectbox(
                "向量重排算法",
                options=["llm", "tfidf", "adaptive"],
                index=["llm", "tfidf", "adaptive"].index(st.session_state.retrieval_reranker),
                format_func=lambda item: RERANKER_LABELS[item],
            )
            st.session_state.retrieval_fusion = st.selectbox(
                "混合融合算法",
                options=["rrf", "weighted_average"],
                index=["rrf", "weighted_average"].index(st.session_state.retrieval_fusion),
                format_func=lambda item: FUSION_LABELS[item],
            )
            st.toggle(
                "自动动态权重",
                key="auto_hybrid_weights",
            )
            st.session_state.vector_weight = st.slider(
                "混合权重：向量 vs BM25",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.vector_weight),
                step=0.05,
                disabled=st.session_state.auto_hybrid_weights,
            )
            weight_text = (
                "自动"
                if st.session_state.auto_hybrid_weights
                else f"BM25 {1.0 - st.session_state.vector_weight:.2f} / 向量 {st.session_state.vector_weight:.2f}"
            )
            st.caption(
                f"当前策略：{MODE_LABELS[st.session_state.retrieval_mode]} | "
                f"权重 {weight_text} | "
                f"{MATCH_TYPE_LABELS[st.session_state.retrieval_match_type]} | "
                f"{RERANKER_LABELS[st.session_state.retrieval_reranker]} | "
                f"{FUSION_LABELS[st.session_state.retrieval_fusion]}"
            )
            if st.session_state.answer_language not in {"zh", "en"}:
                st.session_state.answer_language = "zh"
            st.session_state.answer_language = st.selectbox(
                "回答语言",
                options=["zh", "en"],
                index=["zh", "en"].index(st.session_state.answer_language),
                format_func=lambda language: LANGUAGE_LABELS[language],
            )

            st.divider()
            show_short_memory = st.toggle(
                "💬 短期记忆（最近20条）",
                key="show_short_memory",
            )
            if show_short_memory:
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

            show_long_term_memory = st.toggle(
                "📚 长期记忆",
                key="show_long_term_memory",
            )
            if show_long_term_memory:
                self.long_term_memory.refresh_status()
                memories = self.long_term_memory.list_memories(limit=Config.LONG_TERM_MEMORY_LIMIT)
                if not memories:
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
        if helpful:
            saved = self.long_term_memory.add(last["question"], last["answer"])
            if saved:
                st.success("已存入长期记忆，下次类似问题会优先使用")
            else:
                st.warning("未能写入长期记忆文件，请检查当前目录写入权限。")
        self.rag_engine.record_feedback(
            question=last["question"],
            answer=last["answer"],
            helpful=helpful,
            question_type=last.get("question_type", "entity"),
            source_chunks=last.get("source_chunks"),
        )
        st.success("已记录反馈。")

    def current_retrieval_config(self):
        vector_weight = float(st.session_state.vector_weight)
        hybrid_config = {"fusion": st.session_state.retrieval_fusion}
        if not st.session_state.auto_hybrid_weights:
            hybrid_config.update(
                {
                    "bm25_weight": 1.0 - vector_weight,
                    "vector_weight": vector_weight,
                }
            )
        return {
            "mode": st.session_state.retrieval_mode,
            "bm25": {"match_type": st.session_state.retrieval_match_type},
            "vector": {"reranker": st.session_state.retrieval_reranker},
            "hybrid": hybrid_config,
        }

    def render_chat(self):
        st.title("PDF招股说明书智能问答系统")
        st.caption("当前系统支持 Graph RAG、BM25、向量检索与 RRF 融合；图谱关系用于补充实体、指标、组织与策略类问题的上下文。")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    with st.expander("调试信息"):
                        query_analysis = msg["metadata"].get("query_analysis", {})
                        if "bm25_weight" in query_analysis:
                            st.write(
                                f"当前权重: 关键词 {query_analysis['bm25_weight']} / "
                                f"向量 {query_analysis.get('vector_weight', 0.5)}"
                            )
                        if query_analysis.get("graph_matched_entities"):
                            st.write("图谱命中实体：", "、".join(query_analysis.get("graph_matched_entities", [])))
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
            status_placeholder = st.empty()
            try:
                status_placeholder.info("🔍 正在检索相关文档...")
                time.sleep(0.05)
                status_placeholder.info("🤖 正在生成答案...")
                response = self.rag_engine.ask(
                    prompt,
                    retrieval_mode=st.session_state.retrieval_mode,
                    session_id=st.session_state.session_id,
                    answer_language=st.session_state.answer_language,
                    retrieval_config=self.current_retrieval_config(),
                )
                answer = response.answer
                if response.query_analysis.get("auto_fallback"):
                    st.info(response.query_analysis.get("fallback_notice", "精确匹配无结果，已自动使用模糊匹配"))
                st.markdown(answer)
                status_placeholder.success("✅ 回答完成")

                metadata = {
                    "question_type": response.question_type,
                    "retrieval_mode": response.retrieval_mode,
                    "response_time": response.response_time,
                    "accuracy": response.accuracy,
                    "answer_source": "long_term_memory" if response.memory_hit else "retrieval",
                    "retrieved_count": len(response.retrieved_contexts),
                    "scores": response.scores,
                    "query_analysis": response.query_analysis,
                    "retrieval_config": self.current_retrieval_config(),
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

                graph_info = response.query_analysis.get("graph_rag", {})
                if graph_info:
                    with st.expander("查看知识图谱命中"):
                        matched = graph_info.get("matched_entities", [])
                        if matched:
                            st.write("命中实体：", "、".join(matched))
                        relations = graph_info.get("relations", [])
                        if relations:
                            st.dataframe(relations, use_container_width=True)
                        if self.rag_engine is not None and hasattr(self.rag_engine, "graph_store"):
                            try:
                                st.graphviz_chart(self.rag_engine.graph_store.graphviz_dot_for_query(prompt, limit=24))
                            except Exception:
                                st.info("当前环境未渲染 Graphviz 图，但图谱关系数据已生成。")

                st.session_state.last_response = {
                    "question": prompt,
                    "answer": answer,
                    "question_type": response.question_type,
                    "accuracy": response.accuracy,
                    "source_chunks": response.source_chunks,
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
            retrieval_mode=st.session_state.retrieval_mode,
            retrieval_config=self.current_retrieval_config(),
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
