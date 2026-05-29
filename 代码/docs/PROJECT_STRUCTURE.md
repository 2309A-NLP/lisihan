# 项目结构说明

这个项目现在按“前端页面”和“后端 API”分开组织。后端仍然可以托管前端页面，方便本地一键启动；但代码目录已经分清楚，后续也可以把 `frontend/` 单独部署到 Nginx、对象存储或前端服务器。

## 根目录

- `main.py`：FastAPI 应用入口。负责初始化日志、注册 API 路由、启动 Redis/Milvus/RAG，并托管 `frontend/` 页面。
- `requirements.txt`：Python 依赖。
- `README.md`、`ARCHITECTURE.md`：项目说明和架构文档。
- `logs/`：运行日志目录，包含 `app.log` 和 `error.log`。
- `tests/`：自动化测试。

## frontend/

前端页面目录，只放网页、样式和浏览器端脚本。

- `frontend/index.html`：主聊天页面。
- `frontend/login.html`：登录/注册页面。
- `frontend/admin.html`：管理员控制台页面。
- `frontend/download.html`：对话和知识库下载页面。
- `frontend/assets/style.css`：前端样式文件。
- `frontend/assets/script.js`：旧版或备用聊天脚本。

前端调用后端接口的位置主要在 `index.html`、`admin.html`、`login.html` 内的 `fetch()`。

## app/api/

后端 API 接口层，负责接收前端 HTTP 请求。

- `chat.py`：聊天、流式聊天、会话列表、历史消息接口。
- `user.py`：注册、登录、用户相关接口。
- `role.py`：角色列表、创建角色接口。
- `knowledge.py`：知识库添加、上传、刷新、状态接口。
- `admin.py`：管理员统计、用户管理、会话管理接口。
- `download.py`：下载对话和知识库数据接口。

## app/services/

业务服务层，放具体业务逻辑。

- `role_service.py`：角色定义、角色创建、提示词模板读取。
- `user_service.py`：用户注册、登录、密码校验。
- `knowledge_service.py`：知识库文件解析、文本分块、向量化、写入 Milvus 和缓存。

## app/core/

核心能力层。

- `rag.py`：RAG 核心。负责知识检索、上下文重排、prompt 构造、大模型调用和兜底回答。
- `vectorstore.py`：Milvus 知识库向量存储，负责集合创建、插入、检索。
- `memory.py`：短期记忆和长期记忆。短期记忆用 Redis，长期记忆用 Milvus。
- `logging.py`：日志配置和 traceback 异常日志工具。
- `auth.py`：JWT 鉴权相关逻辑。
- `cleaner.py`：知识文本清洗工具。
- `evaluation.py`：RAG 评测相关逻辑。

## app/models/

数据库模型层。

- `user.py`：用户表模型。
- `role.py`：角色、模板、知识库表模型。
- `chat.py`：会话和消息表模型。
- `__init__.py`：数据库连接、Session 和 Base。

## app/templates/

角色提示词模板。

- `app/templates/roles/医生.txt`
- `app/templates/roles/律师.txt`
- `app/templates/roles/金融理财师.txt`
- 其他角色模板文件

## app/knowledge/

知识库数据和缓存。

- `app/knowledge/data/`：原始知识库文件。
- `app/knowledge/cache/`：解析、分块后的本地缓存，避免每次启动重复处理。

## config/

项目配置。

- `config.py`：数据库、Redis、Milvus、模型等配置。
- `settings.py`、`mysql_config.py`：兼容或拆分配置。

## scripts/

脚本工具。

- `init_db.py`：初始化数据库。
- 其他脚本：检查数据库、重建集合、测试连接等。

## 在线流程

用户在 `frontend/index.html` 输入问题，前端调用 `/api/chat/stream`。后端进入 `app/api/chat.py`，读取角色、历史、短期记忆和长期记忆，再调用 `app/core/rag.py` 检索知识库并生成回答，最后保存消息并返回给前端。

## 离线流程

知识库文件放入 `app/knowledge/data/` 后，`app/services/knowledge_service.py` 会解析文件、分块、向量化，并通过 `app/core/vectorstore.py` 写入 Milvus，同时保存缓存到 `app/knowledge/cache/`。
