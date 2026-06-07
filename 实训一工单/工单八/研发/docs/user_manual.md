# 工单八用户手册

## 启动系统

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

浏览器打开：

```text
http://127.0.0.1:8502
```

## 上传或准备 PDF

将 PDF 放入：

```text
data/
```

系统启动或点击“重新构建索引”后，会解析 PDF 并构建：

- 文本检索索引
- Graph RAG 知识图谱
- Neo4j 图谱数据

## 提问方式

在页面底部文字输入框输入问题，例如：

```text
平安银行法定代表人是谁
```

推荐检索模式：

```text
Graph RAG + 混合检索
```

也可以选择：

- 纯图谱检索
- Graph RAG
- 混合检索
- BM25 检索
- 向量检索

## 查看答案和来源

系统回答后，可查看：

- 答案
- 响应时间
- 置信度
- 检索模式
- 检索片段
- PDF 来源页码

## 查看知识图谱命中

展开：

```text
查看知识图谱命中
```

可看到：

- 命中实体
- 命中关系
- 图谱子图
- 关系来源 PDF

## 查看 Neo4j 可视化

打开：

```text
http://127.0.0.1:7474
```

登录：

```text
账号：neo4j
密码：graphrag2024
```

推荐使用子图查询，不要直接查询全库：

```cypher
MATCH p=(c:Company)-[r]-(n)
WHERE c.name CONTAINS '平安银行'
RETURN p
LIMIT 60;
```

## 运行工单八评估

执行：

```powershell
.\.venv\Scripts\python.exe test_work_order_8.py
```

输出：

```text
output/work_order_8_results.json
output/work_order_8_report.md
```

评估结果包含：

- 每个问题的答案
- 每个问题的图谱实体
- 每个问题的图谱关系
- 每个问题的子图节点和边
- 与工单七结果的对比

## 中英文问答

侧边栏可选择回答语言：

- 中文
- English

## 语音输入说明

当前版本主要支持文字输入。语音输入为可选增强项，如需实现，可在 Streamlit 页面中增加语音转文字组件后复用现有问答接口。
