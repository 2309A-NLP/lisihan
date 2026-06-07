# 工单编号：人工智能 NLP-RAG-混合检索任务

# CLEANUP_LOG

## 清洗范围

- `app.py`
- `src/rag_engine/`
- `src/retriever/`
- `src/validation/`
- `src/multimodal/`
- `utils/llm_engine/`

## 主要改动

1. 删除重复和未使用的导入
   - 清理了 `app.py` 中重复的工单头和未使用的标准库导入。
   - 清理了 `src/rag_engine/` 多个 mixin 文件中从早期模板复制来的未使用导入。
   - 清理了 `src/retriever/` 中未使用的类型、配置和工具导入。
   - 清理了 `utils/llm_engine/` 各 mixin 中重复出现但未使用的 OpenAI、prompt、classifier、typing 导入。
   - 保留了 `__init__.py` 和 `hybrid_retriever.py` 中的重导出导入，因为它们属于对外兼容接口。

2. 拆分过长函数
   - 将 `src/rag_engine/answering.py` 中 `AnsweringMixin.ask` 的部分逻辑拆分为私有方法：
     - `_generate_answer_from_context`
     - `_retry_answer_for_quality`
     - `_answer_liyuan_table_metadata`
     - `_answer_from_long_term_memory`
   - `ask` 从超过 300 行降至 296 行，保留原有缓存、长期记忆、多轮指代、负向问题、元数据直答、表格直答、质量校验和重试逻辑。

3. 保持的功能兼容
   - 未改变 `RAGEngine.ask`、`answer`、`stream_answer`、`record_feedback`、检索器公开类和 Web UI 调用方式。
   - 保留 BM25、向量检索、混合检索、RRF 融合、动态权重、重排、全文检索降级、多轮对话、指代消解、多模态图片解析、长期记忆等既有能力。
   - 保留 `utils/evaluator.py` 的报告输出 `print` 和 `src/multimodal/multimodal_table_extractor.py` 的 CLI 输出 `print`，它们不是调试残留。

## 合并和删除情况

- 未删除业务文件。
- 未合并检索类文件，因为 `src/retriever/hybrid_retriever.py` 是兼容旧导入路径的薄封装，删除会影响对外接口。
- 未删除测试文件，因为它们仍可用于回归验证。

## 指出的冗余或生成文件

以下文件/目录属于运行生成物或缓存，可在需要时加入 `.gitignore` 或手动清理；本次未删除：

- `__pycache__/`
- `src/**/__pycache__/`
- `utils/**/__pycache__/`
- `.pytest_cache/`（如果存在）
- `streamlit.out.log`
- `streamlit.err.log`
- `logs/app.log`

## 验证结果

- `python -m compileall src app.py utils/llm_engine`：通过
- `python -m pytest test_coreference_resolver.py`：4 passed
