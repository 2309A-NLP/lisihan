# 工单八技术文档

## 项目名称

人工智能 NLP-RAG-基于 Graph RAG 实现金融问答。

## 系统架构

系统由 PDF 解析层、混合检索层、知识图谱层、Graph RAG 问答层、评估层和 Streamlit 交互层组成。

- PDF 解析层：读取 `data/` 下 PDF，并复用 `mineru_output/*_chunks.json` 中的文本块。
- 混合检索层：使用 BM25、向量检索和 RRF 融合召回文本片段。
- 知识图谱层：从文本块中抽取公司、人员、组织、财务指标、策略等实体，并抽取实体关系。
- Graph RAG 层：根据问题命中实体，从 Neo4j 或内存图谱中提取相关子图，将子图转换为 LLM 可理解的文本上下文。
- 问答层：支持 `hybrid_graph` 和 `graph_only` 两种 Graph RAG 模式。
- 评估层：读取 `eval_questions.md`，输出每题答案、来源、图谱实体、关系、子图节点和边。

## 技术选型

- Neo4j：持久化知识图谱，地址 `bolt://127.0.0.1:7687`。
- 内存图谱：Neo4j 不可用时自动回退，保证系统可运行。
- Streamlit：用户交互界面。
- BM25 + 向量检索：保留工单七混合检索能力。
- Graph RAG：在文本召回基础上增加实体关系子图上下文。

## 实体与关系创建思路

实体类型：

- Company：公司，如平安银行股份有限公司。
- Person：人员，如董事长、法定代表人、高级管理人员。
- Organization：部门或组织，如分公司、董事会、机构与交易业务委员会。
- Metric：财务指标，如营业收入、净利润、注册资本、信用减值损失。
- Strategy：策略，如哑铃型、资产配置策略、绿色金融、科技金融。
- Value：指标值，如 1379.58 亿元、70.51%。

关系类型：

- `人员`：公司与人员关系。
- `指标`：公司与财务指标数值关系。
- `增长`：公司与同比增长数据关系。
- `设有`：公司与组织、部门、分公司关系。
- `调整`：公司与组织架构调整关系。
- `采用`：公司与策略关系。

所有关系均记录：

- `source_file`
- `page`
- `chunk_id`
- `evidence`
- `confidence`

## Graph RAG 检索流程

1. 用户输入问题。
2. 系统识别问题中的公司、人员、指标、组织等实体。
3. 从 Neo4j 中查询相关一跳或多跳子图。
4. 将图谱关系转换为文本：

```text
[知识图谱关系] 平安银行股份有限公司 --管理人员--> 谢永林
[来源] xxx.pdf 第2页 chunk=p2_005
[原文证据] ...
```

5. 同时执行混合检索。
6. 将图谱上下文和文本上下文合并后生成答案。

## 运行方式

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

启动系统：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

运行工单八评估：

```powershell
.\.venv\Scripts\python.exe test_work_order_8.py
```

输出文件：

- `output/work_order_8_results.json`
- `output/work_order_8_report.md`

## 与工单七对比

工单七使用 Hybrid RAG，只输出文本检索结果。

工单八使用 Graph RAG，除答案外，还输出：

- 命中实体
- 图谱关系
- 子图节点
- 子图边
- PDF 来源页码

## 可视化

Neo4j Browser 地址：

```text
http://127.0.0.1:7474
```

推荐查询：

```cypher
MATCH p=(c:Company)-[r]-(n)
WHERE c.name CONTAINS '平安银行'
RETURN p
LIMIT 60;
```

更多查询见 `neo4j_queries.md`。
