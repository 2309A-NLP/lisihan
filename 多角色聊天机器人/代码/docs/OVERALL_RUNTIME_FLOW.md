# 项目整体运行流程图

这张图按“系统启动 -> 前端访问 -> API 分发 -> 聊天/RAG/知识库/存储”的实际运行路径整理。

```mermaid
flowchart TB
    Start([启动项目<br/>start.bat / python main.py / uvicorn main:app])

    Start --> Main[main.py<br/>创建 FastAPI 应用]
    Main --> Log[初始化日志<br/>app.core.logging.setup_logging]
    Main --> CORS[配置 CORS]
    Main --> Static[托管前端静态文件<br/>frontend/*.html / assets]
    Main --> Routers[注册 API 路由]

    Routers --> ChatRouter["/api/chat<br/>聊天与会话"]
    Routers --> UserRouter["/api/user<br/>注册登录"]
    Routers --> RoleRouter["/api/role<br/>角色管理"]
    Routers --> KnowledgeRouter["/api/knowledge<br/>知识库管理"]
    Routers --> AdminRouter["/api/admin<br/>后台管理"]
    Routers --> DownloadRouter["/api/download<br/>数据下载"]

    Main --> Startup[FastAPI startup 初始化]
    Startup --> RedisInit[init_redis<br/>连接 Redis 短期记忆]
    Startup --> MilvusInit[init_milvus<br/>连接/创建 Milvus 知识集合]
    Startup --> RAGInit[init_rag<br/>加载模型、默认知识、文件知识、BM25 索引]
    Startup --> DBInit[Base.metadata.create_all<br/>检查/创建 MySQL 表]
    DBInit --> Ready([服务就绪<br/>浏览器访问 / /login /admin /download])

    User([用户/管理员]) --> Browser[浏览器前端]
    Ready --> Browser

    Browser --> Index[index.html<br/>聊天页面]
    Browser --> Login[login.html<br/>登录注册]
    Browser --> Admin[admin.html<br/>后台管理]
    Browser --> Download[download.html<br/>下载页面]

    Login --> UserAPI[app/api/user.py]
    UserAPI --> Auth[app/core/auth.py<br/>密码校验/JWT]
    UserAPI --> UserService[app/services/user_service.py]
    UserService --> MySQL[(MySQL<br/>用户/角色/会话/消息)]

    Index --> ChatAPI[app/api/chat.py<br/>POST /api/chat 或 /api/chat/stream]
    ChatAPI --> ValidateRole[RoleService<br/>读取角色与提示词模板]
    ValidateRole --> Templates[(app/templates/roles<br/>角色提示词)]
    ValidateRole --> RoleDB[(MySQL<br/>roles/templates)]

    ChatAPI --> Conversation[校验或创建 Conversation]
    Conversation --> MySQL
    ChatAPI --> History[读取当前会话历史]
    History --> MySQL
    ChatAPI --> ShortMemory[读取短期记忆]
    ShortMemory --> Redis[(Redis<br/>最近对话缓存)]
    ChatAPI --> LongMemory[读取长期记忆]
    LongMemory --> MemoryMilvus[(Milvus<br/>长期对话记忆)]

    ChatAPI --> Context[合并历史 + 短期记忆 + 长期记忆<br/>按预算裁剪去重]
    Context --> RAG[app/core/rag.py<br/>RAG 生成流程]

    RAG --> KnowledgeSearch[混合检索知识<br/>BM25 + 关键词 + Milvus 向量]
    KnowledgeSearch --> KnowledgeMilvus[(Milvus<br/>chatbot_knowledge 知识向量)]
    KnowledgeSearch --> BM25[(本地 BM25 索引<br/>rag_system.knowledge_cache)]
    KnowledgeSearch --> Rerank[相关性排序/去重]
    Rerank --> Prompt[组合角色模板 + 知识片段 + 会话上下文]
    Prompt --> LLM[在线大模型 API<br/>DeepSeek/Chat Completion]
    LLM --> Answer[生成回答]

    Answer --> SaveChat[保存聊天结果]
    SaveChat --> MySQL
    SaveChat --> Redis
    SaveChat --> MemoryMilvus
    SaveChat --> Response[返回前端<br/>JSON 或 SSE: start/meta/token/done]
    Response --> Browser

    Admin --> KnowledgeAPI[app/api/knowledge.py]
    KnowledgeAPI --> Upload[上传/刷新/检查知识库]
    Upload --> SaveFile[(app/knowledge/data<br/>PDF/DOCX/TXT 原始文件)]
    Upload --> KnowledgeService[app/services/knowledge_service.py]
    KnowledgeService --> Parser[app/utils/file_parser.py<br/>按扩展名解析文件]
    Parser --> DOCX[DOCX<br/>python-docx 或 zip+xml]
    Parser --> PDF[PDF<br/>PyMuPDF -> PyPDF2 -> OCR]
    Parser --> TXT[TXT<br/>utf-8/utf-8-sig/gbk]
    DOCX --> Split[文本分块<br/>默认约 1000 字]
    PDF --> Split
    TXT --> Split
    Split --> Embed[rag_system.get_embedding<br/>生成向量]
    Embed --> VectorStore[app/core/vectorstore.py<br/>写入 Milvus]
    VectorStore --> KnowledgeMilvus
    Split --> Cache[(app/knowledge/cache<br/>knowledge_index / role cache)]
    Split --> RebuildBM25[重建 BM25 索引]
    RebuildBM25 --> BM25

    Admin --> AdminAPI[app/api/admin.py]
    Download --> DownloadAPI[app/api/download.py]
    AdminAPI --> MySQL
    AdminAPI --> KnowledgeMilvus
    DownloadAPI --> MySQL
    DownloadAPI --> SaveFile
```

## 聊天请求详细时序

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant FE as 前端 index.html
    participant API as app/api/chat.py
    participant Role as RoleService
    participant DB as MySQL
    participant Mem as memory.py (Redis/Milvus)
    participant RAG as rag.py
    participant VS as vectorstore.py (Milvus)
    participant LLM as 在线大模型

    U->>FE: 输入问题并发送
    FE->>API: POST /api/chat/stream 或 /api/chat
    API->>Role: 获取角色信息和角色提示词
    Role-->>API: role + role_template
    API->>DB: 校验或创建 conversation
    API->>DB: 查询当前会话历史消息
    API->>Mem: 查询短期记忆和长期记忆
    Mem-->>API: 返回相关上下文
    API->>API: 合并上下文并控制长度预算
    API->>RAG: 传入用户问题、角色模板、上下文
    RAG->>VS: 按 role_id 检索知识片段
    VS-->>RAG: 返回 Milvus/BM25 检索结果
    RAG->>RAG: 去重、排序、构建 Prompt
    RAG->>LLM: 请求模型生成回答
    LLM-->>RAG: 返回完整文本或流式 token
    RAG-->>API: 返回回答
    API->>DB: 保存用户消息和助手回复
    API->>Mem: 写入 Redis 短期记忆和 Milvus 长期记忆
    API-->>FE: 返回 JSON 或 SSE 流
    FE-->>U: 展示回复
```

## 知识库入库详细流程

```mermaid
flowchart LR
    A[管理员上传文件<br/>/api/knowledge/upload] --> B[knowledge.py<br/>校验扩展名 .pdf/.docx/.txt]
    B --> C[保存到 app/knowledge/data]
    C --> D[KnowledgeService.add_file_knowledge]
    D --> E[FileParser.parse_file]
    E --> F{文件类型}
    F -->|docx| G[python-docx<br/>失败则 zip+xml 解析]
    F -->|pdf| H[PyMuPDF 提取<br/>失败 PyPDF2<br/>仍失败 OCR]
    F -->|txt| I[尝试 utf-8 / utf-8-sig / gbk]
    G --> J[得到纯文本]
    H --> J
    I --> J
    J --> K[按段落切分为知识片段]
    K --> L[绑定 role_id / knowledge_base_id]
    L --> M[生成 embedding]
    M --> N[vectorstore.insert 写入 Milvus]
    N --> O[同步到 rag_system.knowledge_cache]
    O --> P[重建 BM25 索引]
    P --> Q[更新本地 cache 和数据库更新时间]
```

## 核心数据流向

| 数据 | 写入位置 | 读取场景 |
|---|---|---|
| 用户、角色、会话、聊天消息 | MySQL | 登录、角色选择、历史记录、后台管理 |
| 最近对话上下文 | Redis，失败时降级到内存 | 每次聊天时快速补充上下文 |
| 长期对话记忆 | Milvus memory collection | 用户追问、回忆历史、个性化上下文 |
| 知识库片段向量 | Milvus knowledge collection | RAG 检索知识 |
| 知识库原始文件 | `app/knowledge/data` | 刷新/重建知识库 |
| 知识库解析缓存 | `app/knowledge/cache` | 启动时避免重复解析文件 |
| 角色提示词 | `app/templates/roles` + MySQL | 生成回答前构建 Prompt |
| 运行日志 | `logs/app.log` / `logs/error.log` | 排查异常 |
