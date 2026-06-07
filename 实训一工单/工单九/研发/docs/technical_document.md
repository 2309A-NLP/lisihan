# 工单九技术文档

## 项目名称

人工智能 NLP-RAG-Graph RAG 优化任务。

## 模块清单

- `src/rag_engine_optimized.py`：优化版 Graph RAG 引擎。
- `run_sample_questions.py`：读取 `sample_questions.pdf` 并生成问答结果。
- `evaluate_with_ragas.py`：RAGAS/本地兼容评估脚本。
- `test_ragas_result.py`：RAGAS 测试脚本。
- `docs/optimization_plan.md`：优化方案。

## 优化策略

- 不固定答案。
- 匿名银行题不强制映射到具体银行。
- 匿名银行题禁用 Graph 扩展，防止误命中招股说明书图谱内容。
- 银行/保险分析题限制在金融年报集合内检索。
- 查询扩展覆盖董事长致辞、盈利增长、创新商业模式、风险管理、资本结构、绿色金融、科技金融等维度。
- 检索候选规模为 `candidate_k=5`、`final_k=4`。
- 图谱深度为 `graph_depth=1`。

## 评估流程

```powershell
.\.venv\Scripts\python.exe run_sample_questions.py
.\.venv\Scripts\python.exe test_ragas_result.py
```

输出：

```text
output\sample_questions_graph_rag_results.json
output\ragas_test_result.json
```

标准评估：

```powershell
.\.venv\Scripts\python.exe evaluate_with_ragas.py --results output\sample_questions_graph_rag_results.json
```

输出：

```text
output\ragas_evaluation.json
```

## RAGAS 配置

`.env` 中需要提供：

- `LLM_API_KEY`
- `LLM_API_BASE_URL`
- `LLM_MODEL`

评估脚本会自动映射到 OpenAI-compatible 环境变量。

## 验收指标

| 指标 | 目标 |
| --- | --- |
| context_precision | >= 0.8 |
| context_recall | >= 0.9 |
| average_response_time | <= 3 秒 |
