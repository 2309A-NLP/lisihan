# 多角色聊天机器人项目架构图

## 1. 整体架构

```mermaid
flowchart TB
    User[用户/管理员] --> Browser[浏览器前端]

    subgraph Frontend[前端层 frontend]
        Index[index.html<br/>聊天工作台]
        Login[login.html<br/>登录注册]
        Admin[admin.html<br/>后台管理]
        Download[download.html<br/>数据下载]
        Assets[assets<br/>CSS/JS/静态资源]
    end

    Browser --> Index
    Browser --> Login
    Browser --> Admin
    Browser --> Download

    subgraph Backend[后端应用层 FastAPI]
        Main[main.py<br/>应用入口/路由注册/静态托管]

        subgraph API[API 接口层 app/api]
            ChatAPI[chat.py<br/>聊天/流式聊天/会话历史]
            UserAPI[user.py<br/>注册/登录]
            RoleAPI[role.py<br/>角色管理]
            KnowledgeAPI[knowledge.py<br/>知识库管理]
            AdminAPI[admin.py<br/>后台统计]
            DownloadAPI[download.py<br/>数据导出]
        end

        subgraph Service[业务服务层 app/services]
            UserService[user_service.py<br/>用户服务]
            RoleService[role_service.py<br/>角色/模板服务]
            KnowledgeService[knowledge_service.py<br/>文件解析/知识入库]
        end

        subgraph Core[核心能力层 app/core]
            RAG[rag.py<br/>检索增强生成]
            Memory[memory.py<br/>短期/长期记忆]
            VectorStore[vectorstore.py<br/>Milvus 向量库]
            Auth[auth.py<br/>JWT 鉴权]
            Cleaner[cleaner.py<br/>文本清洗]
            Logging[logging.py<br/>日志与异常]
        end

        subgraph Model[数据模型层 app/models]
            UserModel[user.py<br/>用户模型]
            RoleModel[role.py<br/>角色/模板/知识库模型]
            ChatModel[chat.py<br/>会话/消息模型]
            DBInit[__init__.py<br/>数据库连接/Session]
        end
    end

    Frontend -->|HTTP/Fetch/SSE| Main
    Main --> API
    ChatAPI --> RoleService
    ChatAPI --> RAG
    ChatAPI --> Memory
    ChatAPI --> ChatModel
    UserAPI --> UserService
    UserAPI --> Auth
    RoleAPI --> RoleService
    KnowledgeAPI --> KnowledgeService
    KnowledgeService --> Cleaner
    KnowledgeService --> VectorStore
    AdminAPI --> Model
    DownloadAPI --> Model

    subgraph Storage[数据与基础设施]
        MySQL[(MySQL<br/>用户/角色/会话/消息)]
        Redis[(Redis<br/>短期记忆缓存)]
        Milvus[(Milvus<br/>知识向量/长期记忆)]
        KnowledgeFiles[(app/knowledge/data<br/>PDF/DOCX/TXT)]
        CacheFiles[(app/knowledge/cache<br/>解析缓存)]
        Templates[(app/templates/roles<br/>角色提示词模板)]
        Logs[(logs<br/>运行日志/错误日志)]
        LLM[DeepSeek/在线大模型<br/>Chat Completion API]
    end

    DBInit --> MySQL
    Model --> MySQL
    Memory --> Redis
    Memory --> Milvus
    VectorStore --> Milvus
    KnowledgeService --> KnowledgeFiles
    KnowledgeService --> CacheFiles
    RoleService --> Templates
    RAG --> LLM
    Logging --> Logs
```

## 2. 聊天请求架构链路

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant FE as frontend/index.html
    participant API as app/api/chat.py
    participant Role as RoleService
    participant DB as MySQL
    participant Mem as memory.py<br/>Redis/Milvus
    participant RAG as rag.py
    participant VS as vectorstore.py<br/>Milvus
    participant LLM as 在线大模型

    U->>FE: 输入问题并发送
    FE->>API: POST /api/chat/stream<br/>user_id, role_id, message, conversation_id
    API->>Role: 读取角色与提示词模板
    Role-->>API: role_template
    API->>DB: 校验/创建 conversation
    API->>DB: 查询当前 conversation 历史消息
    API->>Mem: 查询短期记忆和长期记忆
    Mem-->>API: 当前会话相关上下文
    API->>RAG: generate/stream_llm(message, context, role_template)
    RAG->>VS: 按 role_id 检索知识库
    VS-->>RAG: 相关知识片段
    RAG->>RAG: 重排知识 + 构建 Prompt
    RAG->>LLM: 调用模型生成回复
    LLM-->>RAG: 流式 token
    RAG-->>API: token chunks
    API-->>FE: SSE: start/meta/token/done
    FE-->>U: 实时显示回复
    API->>DB: 保存用户消息和 AI 回复
    API->>Mem: 写入短期记忆和长期记忆
```

## 3. 知识库入库架构链路

```mermaid
flowchart LR
    AdminUser[管理员/用户] --> FE[前端上传资料]
    FE --> API[app/api/knowledge.py]
    API --> KS[knowledge_service.py]

    subgraph Parse[文件处理]
        Parser[file_parser.py<br/>PDF/DOCX/TXT 解析]
        Cleaner[cleaner.py<br/>文本清洗]
        Chunk[文本分块]
        RoleBind[按 role_id 绑定知识]
    end

    KS --> Parser
    Parser --> Cleaner
    Cleaner --> Chunk
    Chunk --> RoleBind

    RoleBind --> Embed[rag.py<br/>生成 embedding]
    Embed --> VS[vectorstore.py]
    VS --> Milvus[(Milvus<br/>chatbot_knowledge)]
    RoleBind --> Cache[(app/knowledge/cache<br/>知识索引缓存)]
    API --> MySQL[(MySQL<br/>knowledge_bases)]
```

## 4. 数据存储架构

```mermaid
erDiagram
    users ||--o{ conversations : owns
    users ||--o{ chat_messages : sends
    conversations ||--o{ chat_messages : contains
    roles ||--o{ templates : uses
    roles ||--o{ knowledge_bases : binds
    roles ||--o{ conversations : serves

    users {
        int id
        string username
        string password_hash
        string email
    }

    roles {
        int id
        string name
        string description
        string domain
    }

    templates {
        int id
        int role_id
        text template_content
    }

    knowledge_bases {
        int id
        int role_id
        string name
        text content
    }

    conversations {
        int id
        int user_id
        int role_id
        string title
        datetime created_at
        datetime updated_at
    }

    chat_messages {
        int id
        int conversation_id
        int user_id
        int role_id
        string sender
        text content
        datetime created_at
    }
```

## 5. 模块职责速览

| 层级 | 目录/文件 | 主要职责 |
|---|---|---|
| 前端层 | `frontend/` | 页面展示、用户交互、Fetch/SSE 调用后端 |
| 应用入口 | `main.py` | 初始化 FastAPI、注册路由、托管静态页面、启动 RAG/Redis/Milvus |
| API 层 | `app/api/` | 接收 HTTP 请求，组织业务流程，返回 JSON/SSE |
| 服务层 | `app/services/` | 用户、角色、知识库等业务逻辑封装 |
| 核心层 | `app/core/` | RAG、向量检索、记忆系统、鉴权、日志、文本清洗 |
| 模型层 | `app/models/` | SQLAlchemy ORM 模型与数据库连接 |
| 知识层 | `app/knowledge/` | 原始知识文件与解析缓存 |
| 配置层 | `config/` | 数据库、Redis、Milvus、大模型、JWT 等配置 |
| 测试层 | `tests/`、`scripts/` | API 测试、RAG 测试、文件解析测试、JMeter 压测 |

## 6. 部署视图

```mermaid
flowchart TB
    Client[浏览器客户端] -->|HTTP/SSE| App[FastAPI 服务<br/>uvicorn main:app]

    App --> MySQL[(MySQL<br/>业务数据)]
    App --> Redis[(Redis<br/>短期记忆)]
    App --> Milvus[(Milvus<br/>向量数据)]
    App --> LLM[在线大模型 API]
    App --> LocalFS[本地文件系统<br/>知识文件/模板/缓存/日志]

    subgraph LocalFS[本地文件系统]
        F1[frontend 静态页面]
        F2[app/knowledge/data]
        F3[app/knowledge/cache]
        F4[app/templates/roles]
        F5[logs]
    end
```
