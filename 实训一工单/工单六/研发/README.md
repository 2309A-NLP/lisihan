# PDF招股说明书智能问答系统

本项目是一个面向招股说明书 PDF 文档的智能问答系统，核心能力包括 PDF 解析、表格抽取、图片/图表解析、混合检索、RAG 问答、多轮对话指代消解、短期/长期记忆和答案质量校验。

系统以 `main.py` 作为统一入口，Web 端使用 Streamlit 构建，问答主流程由 `src/rag_engine/` 编排，文档解析由 `pdf_parser/` 完成，检索层采用 BM25 与 Milvus 向量检索融合。

## 项目架构

```text
.
├── app.py                         # Streamlit Web 前端页面
├── main.py                        # 统一启动入口，支持 web / cli 两种模式
├── docker-compose.yml             # Redis 与 Milvus 服务配置
├── requirements.txt               # Python 项目依赖
├── data/                          # PDF 原始文件目录
├── mineru_output/                 # MinerU 解析结果、Markdown、图片和中间产物
├── logs/                          # 系统运行日志
├── run_mineru_segmented.py         # 分段调用 MinerU 的辅助脚本
├── run_mineru_wsl.py              # 通过 WSL 调用 MinerU 的辅助脚本
├── test_*.py                      # 功能测试与工单问题测试
├── 技术文档.md                     # 项目技术说明
├── 用户手册.md                     # 使用说明文档
├── pdf_parser/                    # PDF 解析与结构化抽取模块
│   ├── parser.py                  # PDFParser 与 parse_pdf_file 主实现
│   ├── main.py                    # pdf_parser 包入口
│   ├── cli.py                     # PDF 解析命令行入口
│   ├── mineru_backend.py          # MinerU 后端解析适配
│   ├── text_extractor.py          # 本地文本块抽取
│   ├── table_extractor.py         # 表格抽取
│   ├── chunk_merger.py            # 文本、表格块合并与切分
│   ├── metadata_extractor.py      # 公司名称、法定代表人、注册资本等元数据抽取
│   ├── filters.py                 # 页眉、页脚、页码等噪声过滤
│   ├── visual_extractor.py        # PDF 图片和视觉元素抽取
│   ├── visual_detection.py        # 图像/图表区域检测
│   ├── visual_geometry.py         # 视觉元素坐标与路径处理
│   └── visual_render.py           # PDF 页面渲染为图片
├── src/                           # RAG 系统核心代码
│   ├── config.py                  # 全局配置、环境变量、检索参数
│   ├── constants.py               # 常量定义
│   ├── document.py                # 文档对象定义
│   ├── models.py                  # 初始化结果、问答响应等数据模型
│   ├── memory_manager.py          # Redis 短期记忆与 Milvus 长期记忆管理
│   ├── session_manager.py         # 多轮会话历史管理
│   ├── coreference_resolver.py    # 多轮问答指代消解
│   ├── prompts.py                 # Prompt 模板
│   ├── rag_engine/                # RAG 主流程编排
│   │   ├── engine.py              # RAGEngine 聚合入口
│   │   ├── initializer.py         # PDF 解析与知识库初始化
│   │   ├── context.py             # 检索上下文构建
│   │   ├── answering.py           # 答案生成、重试与响应封装
│   │   ├── metadata.py            # 元数据直答与字段匹配
│   │   ├── multimodal.py          # 图片/图表相关问答处理
│   │   └── cache.py               # 答案缓存
│   ├── retriever/                 # 检索层
│   │   ├── hybrid.py              # 混合检索器，BM25 + 向量检索
│   │   ├── fulltext_retriever.py  # 全文检索
│   │   ├── vector.py              # Milvus 向量检索
│   │   ├── fusion.py              # RRF / 加权融合
│   │   ├── reranker.py            # LLM / TF-IDF / 自适应重排
│   │   ├── query.py               # 查询改写、分词、同义词扩展
│   │   ├── cache.py               # 检索缓存
│   │   └── models.py              # 检索命中结构
│   ├── multimodal/                # 多模态能力
│   │   ├── image_parser.py        # 图片内容解析
│   │   └── multimodal_table_extractor.py
│   ├── processing/                # 查询与表格处理
│   │   ├── query_rewriter.py
│   │   └── table_extractor.py
│   ├── validation/                # 答案质量与负向问题处理
│   │   ├── answer_validator.py
│   │   └── negative_handler.py
│   ├── cache/                     # 元数据缓存
│   │   └── metadata_cache.py
│   └── utils/                     # 文本、检索、WSL MinerU 辅助工具
└── utils/                         # 通用工具模块
    ├── logger.py                  # 日志工具
    ├── evaluator.py               # RAG 评估工具
    ├── question_classifier.py     # 问题分类
    └── llm_engine/                # LLM 调用、规则抽取、数值处理与答案生成
```

## 核心流程

```text
用户问题
  ↓
Streamlit 前端 / CLI 入口
  ↓
RAGEngine
  ↓
会话历史读取与指代消解
  ↓
问题分类、查询改写、元数据快速判断
  ↓
BM25 检索 + Milvus 向量检索
  ↓
RRF / 加权融合 + 重排
  ↓
上下文构建、表格/图片/元数据专项处理
  ↓
规则抽取或 LLM 生成答案
  ↓
答案质量校验、必要时重新检索
  ↓
缓存答案、写入会话记忆
  ↓
返回答案、来源片段和评分信息
```

## 模块说明

### 1. Web 与启动层

- `main.py`：项目统一启动入口，启动前会优先切换到项目 `.venv`，并检查 Streamlit、Redis、Milvus、PDF 解析等核心依赖。
- `app.py`：Streamlit 前端，提供索引管理、检索策略配置、问答对话、短期/长期记忆查看、评估结果展示等界面能力。

### 2. PDF 解析层

- 默认解析后端由 `Config.PDF_PARSER_BACKEND` 控制，当前默认值为 `mineru`。
- `pdf_parser.parser.PDFParser` 会从 `data/` 读取 PDF，优先复用 `mineru_output/` 中已有解析结果；没有缓存时会调用解析后端重新生成结构化片段。
- 解析结果包含文本块、表格块、图片索引、文档元数据和来源页码，用于后续检索与答案溯源。

### 3. 检索层

- `src/retriever/hybrid.py` 是主要检索器，内部维护 BM25 索引，并按配置接入向量检索。
- 支持 `hybrid`、`bm25`、`vector` 检索模式。
- 支持短语匹配、布尔匹配、模糊匹配。
- 支持动态权重：专有名词、数值类问题更偏向 BM25，抽象语义问题更偏向向量检索。
- 支持 RRF 或加权平均融合，并可使用 LLM、TF-IDF 或自适应重排。

### 4. RAG 编排层

`src/rag_engine/engine.py` 中的 `RAGEngine` 通过多个 mixin 组合完整问答链路：

- `InitializationMixin`：解析 PDF 并构建知识库索引。
- `ContextMixin`：根据问题构建检索上下文和来源片段。
- `MetadataMixin`：处理公司名称、注册资本、法定代表人等元数据直答。
- `MultimodalMixin`：处理组织结构图、市场增长图等图片相关问题。
- `AnsweringMixin`：进行答案抽取、LLM 生成、质量重试和响应封装。
- `AnswerCacheMixin`：对短时间内重复问题进行缓存。

### 5. 记忆与会话层

- `src/session_manager.py`：保存多轮对话历史，为指代消解提供上下文。
- `src/coreference_resolver.py`：将“它”“该公司”“这个项目”等指代问题改写为完整问题。
- `src/memory_manager.py`：使用 Redis 保存短期对话，使用 Milvus 保存长期优质问答记忆。

### 6. 多模态与表格问答

- `pdf_parser/visual_*` 负责从 PDF 中抽取、检测和渲染图片区域。
- `src/multimodal/image_parser.py` 调用多模态模型解析图片内容。
- 表格内容会被转为结构化块或 Markdown 文本，参与检索、上下文构建和专项答案抽取。

## 数据与依赖

- PDF 原文放在 `data/` 目录。
- MinerU 解析产物放在 `mineru_output/` 目录。
- Redis 默认地址为 `localhost:6379`。
- Milvus 默认地址为 `localhost:19530`。
- 主要配置集中在 `src/config.py`，也可以通过 `.env` 覆盖。
- Python 依赖见 `requirements.txt`。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Redis 与 Milvus：

```bash
docker compose up -d
```

启动 Web 应用：

```bash
python main.py --mode web
```

启动 CLI 演示：

```bash
python main.py --mode cli
```

直接运行 Streamlit：

```bash
streamlit run app.py --server.port 8502 --server.address localhost
```

## 测试

项目包含以下测试脚本：

```text
test_coreference_resolver.py   # 指代消解测试
test_metadata_lookup.py        # 元数据查询测试
test_工单问题.py                # 工单问题回归测试
```

可使用 pytest 运行：

```bash
python -m pytest
```
