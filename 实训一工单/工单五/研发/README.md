# PDF 招股说明书智能问答系统

本项目是一个面向招股说明书 PDF 的智能问答系统。系统会将 `data/` 目录下的 PDF 解析为结构化片段，构建检索索引，并在 Web 页面中基于 RAG 流程回答用户关于公司基本信息、发行情况、财务表格、行业上下游、技术标准、募集资金等问题。

## 1. 项目定位

系统目标是让用户可以直接围绕招股说明书提问，而不需要手工翻阅长篇 PDF。项目主要能力包括：

- PDF 正文、表格、图片索引和元数据解析
- 招股说明书固定字段抽取，例如公司名称、法定代表人、注册资本、注册地址等
- BM25 + Milvus 向量检索 + RRF 融合的混合检索
- 精确问题的规则化抽取和后处理
- 多轮对话中的指代消解
- Redis 短期对话记忆
- Milvus 长期优质问答记忆
- Streamlit Web 问答界面和命令行评估入口

## 2. 总体架构

```text
PDF 文件
  |
  v
pdf_parser/
  |-- 本地解析后端：PyMuPDF + pdfplumber
  |-- 可选 MinerU 后端：WSL/conda 中的 MinerU CLI
  v
mineru_output/
  |-- *_chunks.json
  |-- *_metadata.json
  |-- images/
  v
src.rag_engine.RAGEngine
  |-- 初始化知识库
  |-- 问题分类
  |-- 指代消解
  |-- 检索上下文构建
  |-- 规则抽取 / LLM 生成 / 答案校验
  v
src.retriever.HybridRetriever
  |-- BM25
  |-- Milvus 向量检索
  |-- RRF 融合排序
  v
Streamlit Web / CLI 输出
```

辅助服务：

- Redis：保存短期对话历史
- Milvus：提供向量检索和长期问答记忆
- LLM API：在规则抽取不足时生成最终答案

## 3. 目录结构

```text
.
├── app.py                         # Streamlit Web 页面
├── main.py                        # 统一启动入口，支持 web/cli 两种模式
├── run_mineru_wsl.py              # 可选：批量调用 WSL MinerU 的脚本
├── requirements.txt               # Python 依赖
├── docker-compose.yml             # Redis/Milvus 服务配置
├── data/                          # PDF 原文目录
├── mineru_output/                 # PDF 解析输出和图片索引
├── logs/                          # 运行日志
├── pdf_parser/                    # PDF 解析子系统
├── src/                           # RAG、检索、记忆、配置等核心代码
├── utils/                         # LLM、评估、日志、问题分类等工具
├── 技术文档.md
└── 用户手册.md
```

## 4. 核心模块

### 4.1 启动层

入口文件：

- `main.py`
- `app.py`

`main.py` 负责检查项目虚拟环境和依赖，并根据参数启动 Web 或 CLI：

```powershell
.\.venv\Scripts\python.exe main.py --mode web
.\.venv\Scripts\python.exe main.py --mode cli
```

`app.py` 是 Streamlit 页面，负责展示侧边栏配置、聊天窗口、检索调试信息、短期记忆、长期记忆和批量评估入口。

### 4.2 PDF 解析层

目录：

```text
pdf_parser/
```

主要文件：

- `parser.py`：统一 PDF 解析入口，按配置选择解析后端
- `text_extractor.py`：基于 PyMuPDF 抽取正文
- `table_extractor.py`：基于 pdfplumber 抽取表格
- `visual_extractor.py`：抽取图片和页面视觉索引
- `chunk_merger.py`：合并正文、表格和图片索引，生成 RAG 片段
- `metadata_extractor.py`：抽取公司固定元数据
- `mineru_backend.py`：适配 MinerU 输出
- `cli.py`：命令行解析入口

当前默认解析后端为：

```env
PDF_PARSER_BACKEND=local
```

本地后端会生成项目可直接使用的：

```text
mineru_output/<PDF文件名>_chunks.json
mineru_output/<PDF文件名>_metadata.json
mineru_output/images/
```

### 4.3 RAG 编排层

目录：

```text
src/rag_engine/
```

主要文件：

- `engine.py`：组合各个 mixin，形成 `RAGEngine`
- `initializer.py`：读取 PDF、解析文档、构建知识库索引
- `answering.py`：问答主流程
- `context.py`：构建检索上下文
- `metadata.py`：固定字段元数据问答
- `multimodal.py`：多模态图片/表格问题处理
- `cache.py`：答案缓存

问答流程大致为：

1. 规范化用户问题
2. 根据会话历史做指代消解
3. 判断是否属于图片/多模态问题
4. 判断是否属于否定类或固定字段问题
5. 查询长期记忆
6. 构建检索上下文
7. 对实体、数值、比例类问题优先做规则抽取
8. 必要时调用 LLM 生成答案
9. 做答案质量校验和后处理
10. 写入会话历史与缓存

### 4.4 检索层

目录：

```text
src/retriever/
```

主要文件：

- `hybrid.py`：混合检索主类 `HybridRetriever`
- `query.py`：查询改写、分词、关键词增强
- `vector.py`：Milvus 向量检索
- `fusion.py`：RRF 融合排序
- `cache.py`：检索结果缓存
- `models.py`：检索命中对象定义

检索策略：

- BM25 是默认主召回方式，适合招股说明书中的精确关键词问题
- Milvus 向量检索用于语义问题的补充召回
- RRF 将 BM25 与向量检索结果融合
- 对固定字段、金额、比例、标准、上下游等精确问题，会尽量走更快的规则和 BM25 路径

### 4.5 问题理解与处理层

相关目录和文件：

- `utils/question_classifier.py`
- `src/processing/query_rewriter.py`
- `src/coreference_resolver.py`
- `src/validation/answer_validator.py`
- `src/validation/negative_handler.py`
- `utils/llm_engine/`

职责：

- 将问题分为 `entity`、`numeric`、`percentage` 等类型
- 将自然语言问题改写为更贴近招股说明书原文的检索表达
- 处理“该公司”“它”“上述公司”等多轮指代
- 对否定类问题进行专门处理
- 对生成答案做质量校验
- 对金额、比例、实体、表格答案做后处理

### 4.6 记忆层

文件：

```text
src/memory_manager.py
```

包含两类记忆：

- `RedisMemoryManager`：短期会话记忆，保存最近对话
- `LongTermMemoryManager`：长期优质问答记忆，使用 Milvus 保存被标记为有帮助的问答

相关配置：

```env
REDIS_URL=redis://localhost:6379/0
SHORT_MEMORY_LIMIT=20
LONG_TERM_MEMORY_COLLECTION=long_term_memory
ENABLE_LONG_TERM_MEMORY_ANSWER=false
```

### 4.7 多模态层

相关目录：

```text
src/multimodal/
src/rag_engine/multimodal.py
pdf_parser/visual_*.py
```

主要职责：

- 抽取 PDF 中的图片和页面渲染图
- 建立图片与页码、标题、来源文件之间的索引
- 对组织结构图、市场结构图等图像问题调用多模态模型

## 5. 数据流

### 5.1 离线/初始化阶段

```text
data/*.pdf
  -> pdf_parser.PDFParser.parse_multiple_pdfs()
  -> mineru_output/*_chunks.json
  -> Document(page_content, metadata)
  -> HybridRetriever.create_vectorstore()
  -> BM25 索引
```

如果 Milvus 和 embedding 模型可用，向量检索会在查询阶段作为辅助召回使用。

### 5.2 在线问答阶段

```text
用户问题
  -> 问题分类
  -> 指代消解
  -> 查询改写 / 文档选择
  -> BM25 / Milvus 检索
  -> RRF 融合
  -> 规则抽取或 LLM 生成
  -> 答案校验与后处理
  -> Web 页面展示
```

## 6. 配置说明

主要配置位于：

```text
.env
src/config.py
```

常用配置：

```env
PDF_DIR=./data
PDF_PARSER_BACKEND=local
PDF_PARSE_OUTPUT_DIR=./mineru_output
MINERU_OUTPUT_DIR=./mineru_output

BM25_K=6
VECTOR_K=4
FINAL_K=3
RRF_K=30

REDIS_URL=redis://localhost:6379/0
MILVUS_URI=http://localhost:19530

LLM_PROVIDER=openai
LLM_API_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

PDF 后端说明：

- `local`：默认后端，使用 PyMuPDF/pdfplumber，不依赖 WSL
- `mineru`：可选后端，通过 WSL 调用 MinerU CLI
- `pymupdf`：本地后端别名

## 7. 运行方式

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

可选启动 Redis/Milvus：

```powershell
docker compose up -d
```

启动 Web：

```powershell
.\.venv\Scripts\python.exe main.py --mode web
```

默认访问：

```text
http://localhost:8502
```

运行 CLI 评估：

```powershell
.\.venv\Scripts\python.exe main.py --mode cli
```

手动解析单个 PDF：

```powershell
.\.venv\Scripts\python.exe -m pdf_parser.cli data\招股说明书1.pdf -o mineru_output --backend local --table_to_text --max_text_chars 500
```

## 8. 可选 MinerU 架构

项目保留了 MinerU 适配层，但它不是当前默认运行路径。启用 MinerU 时，流程为：

```text
Windows Python
  -> wsl.exe bash -lc
  -> conda activate <MINERU_CONDA_ENV> 可选
  -> mineru / magic-pdf
  -> MinerU 原始 md 和 content_list.json
  -> mineru_backend.py 适配为项目 chunks JSON
```

相关配置：

```env
PDF_PARSER_BACKEND=mineru
MINERU_BACKEND=pipeline
MINERU_MODEL_SOURCE=modelscope
MINERU_CLI=mineru
MINERU_CONDA_ROOT=~/miniconda3
MINERU_CONDA_ENV=
MINERU_WSL_DISTRO=
```

如果可执行命令不是 `mineru`，例如叫 `magic-pdf`，则设置：

```env
MINERU_CLI=magic-pdf
```

## 9. 输出文件说明

解析输出位于：

```text
mineru_output/
```

关键文件：

- `*_chunks.json`：RAG 使用的正文、表格、图片索引片段
- `*_metadata.json`：公司固定元数据
- `images/`：从 PDF 中抽取或渲染的图片

`*_chunks.json` 中的主要字段：

- `source_file`
- `source_path`
- `page_count`
- `chunk_count`
- `table_count`
- `image_count`
- `parser_backend`
- `document_metadata`
- `visual_index`
- `chunks`

每个 chunk 通常包含：

- `type`：`text`、`table`、`image`
- `page`
- `bbox`
- `content`
- `chunk_id`
- `metadata`

## 10. 日志与排查

日志文件：

```text
logs/app.log
streamlit.err.log
streamlit.out.log
```

常见检查点：

- 知识库为空：检查 `data/` 下是否有 PDF
- 解析失败：检查 `PDF_PARSER_BACKEND` 和 `mineru_output/`
- Redis 不可用：检查 `REDIS_URL` 和 6379 端口
- Milvus 不可用：检查 `MILVUS_URI` 和 19530 端口
- 答案不准确：先在页面展开检索片段，确认命中的上下文是否正确

## 11. 当前验证状态

当前项目已验证：

- `PDF_PARSER_BACKEND=local`
- `招股说明书1.pdf`、`招股说明书2.pdf` 已生成本地解析 chunks
- 知识库初始化成功，索引文档片段数为 `3588`
- Streamlit Web 服务可通过 `http://127.0.0.1:8502` 访问

