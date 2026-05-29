# 基于RAG的多角色聊天机器人项目架构设计

## 1. 项目概述

本项目是一个基于RAG（检索增强生成）技术的多角色聊天机器人系统，支持多种角色的智能对话，包括社交NPC、医生、心理医生、律师、股票分析师、金融理财师、科学家、教师和英语学习助手等。

## 2. 技术栈

### 2.1 核心技术
- **大模型**：支持本地部署（vLLM、SGLang、xInernce）和在线API
- **RAG系统**：检索增强生成
- **向量数据库**：Milvus（支持Linux混合检索BM25）
- **向量化模型**：BGE-m3
- **重排序模型**：BGE-rerank
- **短期记忆**：Redis（保存最近聊天记录）
- **长期记忆**：Milvus
- **关系型数据库**：MySQL（存储用户信息和角色信息）
- **API框架**：FastAPI（高性能异步API）
- **测试工具**：Postman（API测试）、Jmeter（压力测试）
- **RAG评测**：RAGAS

### 2.2 技术选型理由
- **FastAPI**：高性能、自动生成API文档、支持异步处理
- **Milvus**：专为向量检索优化的数据库，支持混合检索
- **Redis**：高性能缓存，适合存储短期记忆
- **MySQL**：成熟的关系型数据库，适合存储结构化数据
- **BGE模型**：中文表现优秀，适合处理中文对话和知识库

## 3. 系统架构

### 3.1 核心模块

1. **用户管理模块**
   - 用户注册、登录、信息管理
   - 用户会话管理

2. **角色管理模块**
   - 角色定义和配置
   - 提示词模板管理
   - 角色知识库管理

3. **RAG核心模块**
   - 知识库检索
   - 向量化处理
   - 重排序
   - 大模型调用

4. **记忆管理模块**
   - 短期记忆（Redis）
   - 长期记忆（Milvus）

5. **API接口模块**
   - 聊天接口
   - 角色管理接口
   - 知识库管理接口

6. **评测和监控模块**
   - RAG性能评测
   - 系统监控

### 3.2 数据流

1. 用户发送消息 → API接口 → 记忆管理（获取历史对话）
2. 消息处理 → RAG核心（检索相关知识）
3. 生成回复 → 记忆管理（存储对话）→ 返回用户

## 4. 目录结构

```
multi-role-chatbot/
├── app/
│   ├── api/                # API接口
│   │   ├── __init__.py
│   │   ├── chat.py         # 聊天接口
│   │   ├── user.py         # 用户管理接口
│   │   ├── role.py         # 角色管理接口
│   │   └── knowledge.py    # 知识库管理接口
│   ├── core/               # 核心模块
│   │   ├── __init__.py
│   │   ├── rag.py          # RAG核心
│   │   ├── vectorstore.py  # 向量存储
│   │   └── memory.py       # 记忆管理
│   ├── models/             # 数据模型
│   │   ├── __init__.py
│   │   ├── user.py         # 用户模型
│   │   └── role.py         # 角色模型
│   ├── services/           # 服务层
│   │   ├── __init__.py
│   │   ├── user_service.py # 用户服务
│   │   ├── role_service.py # 角色服务
│   │   └── chat_service.py # 聊天服务
│   ├── templates/          # 提示词模板
│   │   ├── __init__.py
│   │   └── roles/          # 角色提示词模板
│   ├── knowledge/          # 知识库
│   │   ├── __init__.py
│   │   └── data/           # 知识库数据
│   └── utils/              # 工具函数
│       ├── __init__.py
│       ├── embedding.py    # 向量化工具
│       └── rerank.py       # 重排序工具
├── config/                 # 配置文件
│   ├── __init__.py
│   └── config.py           # 主配置文件
├── scripts/                # 脚本工具
│   ├── __init__.py
│   ├── init_db.py          # 数据库初始化
│   └── load_knowledge.py   # 知识库加载
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── test_rag.py         # RAG测试
│   └── test_api.py         # API测试
├── main.py                 # 应用入口
├── requirements.txt        # 依赖包
└── README.md               # 项目说明
```

## 5. 数据库设计

### 5.1 用户表（users）
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| id | INT | 用户ID（主键） |
| username | VARCHAR(50) | 用户名 |
| password_hash | VARCHAR(255) | 密码哈希 |
| email | VARCHAR(100) | 邮箱 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### 5.2 角色表（roles）
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| id | INT | 角色ID（主键） |
| name | VARCHAR(50) | 角色名称 |
| description | TEXT | 角色描述 |
| template_id | INT | 提示词模板ID |
| knowledge_base_id | INT | 知识库ID |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### 5.3 提示词模板表（templates）
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| id | INT | 模板ID（主键） |
| name | VARCHAR(50) | 模板名称 |
| content | TEXT | 模板内容 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### 5.4 知识库表（knowledge_bases）
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| id | INT | 知识库ID（主键） |
| name | VARCHAR(100) | 知识库名称 |
| description | TEXT | 知识库描述 |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### 5.5 对话表（conversations）
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| id | INT | 对话ID（主键） |
| user_id | INT | 用户ID |
| role_id | INT | 角色ID |
| created_at | DATETIME | 创建时间 |
| updated_at | DATETIME | 更新时间 |

### 5.6 消息表（messages）
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| id | INT | 消息ID（主键） |
| conversation_id | INT | 对话ID |
| sender | VARCHAR(20) | 发送者（user/role） |
| content | TEXT | 消息内容 |
| created_at | DATETIME | 创建时间 |

## 6. API设计

### 6.1 聊天接口
- **POST /api/chat**：发送消息并获取回复
  - 请求体：{"user_id": 1, "role_id": 1, "message": "你好"}
  - 响应：{"response": "你好，有什么可以帮助你的吗？", "conversation_id": 1}

### 6.2 用户管理接口
- **POST /api/user/register**：注册新用户
  - 请求体：{"username": "user1", "password": "password123", "email": "user1@example.com"}
  - 响应：{"user_id": 1, "username": "user1"}

- **POST /api/user/login**：用户登录
  - 请求体：{"username": "user1", "password": "password123"}
  - 响应：{"access_token": "token", "token_type": "bearer"}

### 6.3 角色管理接口
- **GET /api/role/list**：获取角色列表
  - 响应：[{"id": 1, "name": "医生", "description": "医疗健康顾问"}]

- **POST /api/role/create**：创建新角色
  - 请求体：{"name": "教师", "description": "教育顾问", "template_id": 1, "knowledge_base_id": 1}
  - 响应：{"role_id": 2, "name": "教师"}

### 6.4 知识库管理接口
- **POST /api/knowledge/add**：添加知识库内容
  - 请求体：{"knowledge_base_id": 1, "content": "高血压治疗指南..."}
  - 响应：{"status": "success"}

- **POST /api/knowledge/update**：更新知识库内容
  - 请求体：{"knowledge_base_id": 1, "content": "更新后的高血压治疗指南..."}
  - 响应：{"status": "success"}

## 7. 核心流程

### 7.1 聊天流程
1. 用户发送消息到API接口
2. 系统获取用户和角色信息
3. 从Redis获取短期记忆（最近聊天记录）
4. 从Milvus获取长期记忆（相关历史对话）
5. 使用RAG系统检索相关知识库内容
6. 调用大模型生成回复
7. 将对话存储到Redis（短期记忆）和MySQL（持久化）
8. 返回回复给用户

### 7.2 知识库更新流程
1. 管理员通过API上传知识库内容
2. 系统使用BGE-m3模型对内容进行向量化
3. 将向量存储到Milvus
4. 更新知识库元数据到MySQL

## 8. 部署方案

### 8.1 本地开发环境
- Python 3.8+
- FastAPI
- MySQL
- Redis
- Milvus（可使用Docker部署）

### 8.2 生产环境
- 云服务器
- 容器化部署（Docker + Kubernetes）
- 负载均衡
- 监控系统

## 9. 性能优化

### 9.1 RAG优化
- 向量索引优化
- 检索策略优化
- 重排序模型调优

### 9.2 系统优化
- 缓存策略优化
- 异步处理
- 数据库索引优化

## 10. 未来扩展

- 支持更多角色类型
- 多语言支持
- 情感分析
- 个性化推荐
- 多模态支持（图片、语音）
