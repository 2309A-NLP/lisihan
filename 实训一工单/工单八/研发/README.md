# PDF 招股说明书智能问答系统

本项目面向招股说明书、年度报告等金融 PDF 文档，构建了一个支持文档解析、混合检索、Graph RAG、多轮指代消解、结构化问答、图像/表格解析和记忆增强的智能问答系统。系统提供 Streamlit Web 页面和 CLI 两种入口。

## 项目架构图

```mermaid
flowchart TB
    user["用户"] --> ui["Streamlit Web UI<br/>app.py"]
    user --> cli["CLI 启动入口<br/>main.py --mode cli"]
    cli --> engine["RAGEngine<br/>src/rag_engine/engine.py"]
    ui --> engine

    subgraph ingest["知识库构建流程"]
        data["PDF 原始文件<br/>data/"] --> parser["PDFParser<br/>pdf_parser/"]
        parser --> text["正文抽取<br/>text_extractor.py"]
        parser --> table["表格抽取<br/>table_extractor.py"]
        parser --> visual["图片/页面渲染<br/>visual_extractor.py"]
        text --> chunks["分块与合并<br/>chunk_merger.py"]
        table --> chunks
        visual --> mm_assets["多模态素材<br/>mineru_output/images/"]
        chunks --> metadata["元数据抽取<br/>metadata_extractor.py"]
        chunks --> documents["Document Chunks"]
    end

    subgraph retrieval["检索与知识组织"]
        documents --> bm25["BM25 索引<br/>rank_bm25"]
        documents --> vector["向量召回<br/>sentence-transformers / Milvus"]
        documents --> graph_build["实体关系抽取<br/>GraphEntityExtractor"]
        graph_build --> graph_store["知识图谱存储<br/>Neo4j + 内存兜底"]
        bm25 --> hybrid["HybridRetriever<br/>BM25 + Vector + RRF/加权融合"]
        vector --> hybrid
        graph_store --> graph_rag["GraphRAGRetriever"]
    end

    subgraph qa["问答核心链路"]
        engine --> session["会话管理<br/>SessionManager / Redis"]
        engine --> coref["多轮指代消解<br/>CoreferenceResolver"]
        engine --> classifier["问题分类<br/>QuestionClassifier"]
        engine --> rewrite["Query 改写与文档选择<br/>MetadataMixin / ContextMixin"]
        rewrite --> fast_path["元数据/负向问题/表格规则直答"]
        rewrite --> hybrid
        rewrite --> graph_rag
        hybrid --> context["上下文拼装<br/>ContextMixin"]
        graph_rag --> context
        fast_path --> answer["答案生成与后处理<br/>AnsweringMixin / LLMEngine"]
        context --> answer
        answer --> validator["质量校验与必要重试<br/>answer_validator.py"]
        validator --> response["RAGResponse<br/>答案、来源、分数、分析信息"]
    end

    subgraph memory["缓存与记忆"]
        response --> answer_cache["短期答案缓存<br/>RAGEngine 内存缓存"]
        response --> short_memory["短期对话记忆<br/>Redis"]
        response --> long_memory["长期优质问答记忆<br/>memory.json / Milvus"]
        long_memory --> engine
        short_memory --> coref
    end

    subgraph external["外部服务与配置"]
        redis["Redis<br/>localhost:6379"]
        milvus["Milvus<br/>localhost:19530"]
        neo4j["Neo4j<br/>bolt://localhost:7687"]
        llm["LLM / 多模态模型 API"]
        config["系统配置<br/>src/config.py / .env"]
    end

    session -.-> redis
    short_memory -.-> redis
    vector -.-> milvus
    long_memory -.-> milvus
    graph_store -.-> neo4j
    answer -.-> llm
    visual -.-> llm
    config -.-> engine
```

## 核心模块说明

| 模块 | 位置 | 职责 |
| --- | --- | --- |
| Web 前端 | `app.py` | 提供 Streamlit 问答页面、索引管理、检索策略切换、反馈与记忆查看 |
| 启动入口 | `main.py` | 检查依赖，支持 Web/CLI 两种运行模式 |
| PDF 解析 | `pdf_parser/` | 解析 PDF 正文、表格、图片和页面渲染结果，输出文档 chunks 与元数据 |
| RAG 编排 | `src/rag_engine/` | 负责初始化知识库、构建上下文、问答生成、缓存、质量重试和多模态问答 |
| 混合检索 | `src/retriever/` | 提供 BM25、向量检索、动态权重、RRF/加权融合、重排与反馈调权 |
| Graph RAG | `src/graph_rag/` | 从文档抽取实体关系，构建知识图谱，并按问题召回关系证据 |
| 记忆管理 | `src/memory_manager.py` | 管理 Redis 短期对话记忆，以及本地/Milvus 长期优质问答记忆 |
| 多轮理解 | `src/coreference_resolver.py`、`src/session_manager.py` | 结合历史对话完成指代消解和当前公司识别 |
| 答案生成 | `utils/llm_engine/` | 负责规则抽取、LLM 生成、后处理、本地兜底答案 |
| 评估工具 | `utils/evaluator.py`、`test_work_order_*.py` | 支持 RAG 效果评估和工单回归测试 |

## 数据流概览

1. 将 PDF 放入 `data/`。
2. `RAGEngine.initialize_project_knowledge_base()` 调用 `PDFParser` 解析 PDF，生成文档 chunks。
3. 系统基于 chunks 同时构建 BM25 索引、向量召回数据和知识图谱。
4. 用户提问后，系统先进行会话指代消解、问题分类、Query 改写和目标 PDF 选择。
5. 对固定元数据、负向问题、部分表格问题优先走规则直答；其他问题走混合检索或 Graph RAG。
6. 检索结果被拼装为上下文后，进入规则抽取或 LLM 生成，并经过答案后处理和质量校验。
7. 最终返回答案、来源片段、相关分数、检索模式和分析信息，同时更新会话记忆与可选长期记忆。

## 运行入口

```powershell
python main.py --mode web
```

```powershell
python main.py --mode cli
```

如需启用 Redis、Milvus 等服务，可使用项目中的 Docker 配置：

```powershell
docker compose up -d
```
