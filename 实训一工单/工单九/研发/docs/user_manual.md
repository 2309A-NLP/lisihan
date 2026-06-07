# 工单九用户手册

## 1. 生成 sample_questions 问答结果

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe run_sample_questions.py
```

生成：

```text
output\sample_questions_graph_rag_results.json
```

## 2. 运行 RAGAS 测试

```powershell
.\.venv\Scripts\python.exe test_ragas_result.py
```

生成：

```text
output\ragas_test_result.json
```

## 3. 运行标准评估

```powershell
.\.venv\Scripts\python.exe evaluate_with_ragas.py --results output\sample_questions_graph_rag_results.json
```

生成：

```text
output\ragas_evaluation.json
```

## 4. 重新测试

如果要重新测试，先清空 `output/` 目录中的旧文件，再运行：

```powershell
.\.venv\Scripts\python.exe run_sample_questions.py
.\.venv\Scripts\python.exe test_ragas_result.py
```

## 5. 注意事项

- 答案不会固定化。
- 匿名银行题只做来源过滤和关键词扩展。
- 如果 RAGAS 报模型错误，检查 `.env` 中的 `LLM_MODEL`、`LLM_API_BASE_URL`、`LLM_API_KEY`。
- 如果响应时间超过 3 秒，查看 `sample_questions_graph_rag_results.json` 中每题的 `response_time`。
- 如果命中错误 PDF，查看 `source_chunks[].metadata.source_file`。
