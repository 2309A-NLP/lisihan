# 基于RAG的多角色聊天机器人项目

## 项目概述

本项目是一个基于RAG（检索增强生成）技术的多角色聊天机器人系统，支持多种角色的智能对话，包括社交NPC、医生、心理医生、律师、股票分析师、金融理财师、科学家、教师和英语学习助手等。

## 技术栈

- **后端框架**：FastAPI
- **数据库**：MySQL、Redis、Milvus
- **大模型**：支持本地部署（vLLM、SGLang、xInernce）和在线API
- **向量化模型**：BGE-m3
- **重排序模型**：BGE-rerank
- **RAG评测**：RAGAS
- **测试工具**：Postman、Jmeter

## 项目结构

```
multi-role-chatbot/
├── app/                # 应用代码
│   ├── api/            # API接口
│   │   ├── chat.py     # 聊天接口
│   │   ├── user.py     # 用户管理接口
│   │   ├── role.py     # 角色管理接口
│   │   └── knowledge.py # 知识库管理接口
│   ├── core/           # 核心模块
│   │   ├── rag.py      # RAG核心
│   │   ├── vectorstore.py # 向量存储
│   │   └── memory.py   # 记忆管理
│   ├── models/         # 数据模型
│   │   ├── user.py     # 用户模型
│   │   └── role.py     # 角色模型
│   ├── services/       # 服务层
│   │   ├── user_service.py # 用户服务
│   │   ├── role_service.py # 角色服务
│   │   └── knowledge_service.py # 知识库服务
│   ├── templates/      # 提示词模板
│   │   └── roles/      # 角色提示词模板
│   ├── knowledge/      # 知识库
│   │   └── data/       # 知识库数据
│   └── utils/          # 工具函数
├── config/             # 配置文件
│   └── config.py       # 主配置文件
├── scripts/            # 脚本工具
│   ├── init_db.py      # 数据库初始化
│   └── load_knowledge.py # 知识库加载
├── tests/              # 测试代码
├── main.py             # 应用入口
├── requirements.txt    # 依赖包
└── README.md           # 项目说明
```

## 目录和文件说明

### app/api/
- **chat.py**：聊天接口，处理用户消息并返回角色回复
- **user.py**：用户管理接口，处理用户注册和登录
- **role.py**：角色管理接口，处理角色的创建和查询
- **knowledge.py**：知识库管理接口，处理知识库的添加和更新

### app/core/
- **rag.py**：RAG核心模块，实现检索增强生成的逻辑
- **vectorstore.py**：向量存储模块，管理Milvus向量数据库
- **memory.py**：记忆管理模块，管理Redis短期记忆和Milvus长期记忆

### app/models/
- **user.py**：用户数据模型
- **role.py**：角色、模板和知识库数据模型

### app/services/
- **user_service.py**：用户服务，处理用户注册、登录和密码验证
- **role_service.py**：角色服务，处理角色管理和提示词模板
- **knowledge_service.py**：知识库服务，处理知识库的管理和更新

### app/templates/roles/
- 存放角色提示词模板文件

### app/knowledge/data/
- 存放知识库数据文件

### config/
- **config.py**：项目配置文件，包含数据库连接、模型配置等

### scripts/
- **init_db.py**：数据库初始化脚本，创建表结构和初始化基础数据
- **load_knowledge.py**：知识库加载脚本，将数据加载到向量数据库

## 快速开始

### 1. 环境准备

- Python 3.8+
- MySQL
- Redis
- Milvus（可使用Docker部署）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置修改

编辑 `config/config.py` 文件，修改以下配置：

- `DATABASE_URL`：MySQL数据库连接URL
- `REDIS_HOST`、`REDIS_PORT`：Redis连接配置
- `MILVUS_HOST`、`MILVUS_PORT`：Milvus连接配置
- `MODEL_TYPE`、`API_KEY`：大模型配置

### 4. 初始化数据库

```bash
python scripts/init_db.py
```

### 5. 启动服务

```bash
uvicorn main:app --reload
```

服务将在 `http://localhost:8000` 启动，API文档可访问 `http://localhost:8000/docs`。

## API使用

### 1. 用户注册

**POST /api/user/register**

```json
{
  "username": "user1",
  "password": "password123",
  "email": "user1@example.com"
}
```

### 2. 用户登录

**POST /api/user/login**

```json
{
  "username": "user1",
  "password": "password123"
}
```

### 3. 获取角色列表

**GET /api/role/list**

### 4. 聊天

**POST /api/chat**

```json
{
  "user_id": 1,
  "role_id": 1,
  "message": "你好"
}
```

### 5. 添加知识库内容

**POST /api/knowledge/add**

```json
{
  "knowledge_base_id": 1,
  "content": "高血压治疗指南..."
}
```

## 角色说明

本系统支持以下角色：

1. **医生**：医疗健康顾问
2. **心理医生**：心理咨询师
3. **律师**：法律顾问
4. **股票分析师**：股票投资顾问
5. **金融理财师**：理财规划顾问
6. **科学家**：科学顾问
7. **教师**：教育顾问
8. **英语学习助手**：英语学习顾问
9. **虚拟朋友**：虚拟社交伙伴

## 知识库说明

系统包含以下知识库：

1. **医疗知识库**：包含医疗健康相关知识
2. **心理学知识库**：包含心理学相关知识
3. **法律知识库**：包含法律法规相关知识
4. **金融知识库**：包含金融投资相关知识
5. **科学知识库**：包含科学技术相关知识
6. **教育知识库**：包含教育教学相关知识
7. **英语知识库**：包含英语学习相关知识
8. **社交知识库**：包含社交交流相关知识

## 性能优化

1. **RAG优化**：
   - 向量索引优化
   - 检索策略优化
   - 重排序模型调优

2. **系统优化**：
   - 缓存策略优化
   - 异步处理
   - 数据库索引优化

## 压力测试

项目已提供 JMeter 压力测试计划：

- 测试计划：`tests/jmeter/chat_load_test.jmx`
- 参数文件：`tests/jmeter/chat_questions.csv`
- Windows 运行脚本：`scripts/run_jmeter_load_test.bat`
- 使用说明：`docs/JMETER_LOAD_TEST.md`

启动服务后运行：

```bat
scripts\run_jmeter_load_test.bat
```

报告会生成到 `tests/jmeter/results/report/index.html`。

## 未来扩展

- 支持更多角色类型
- 多语言支持
- 情感分析
- 个性化推荐
- 多模态支持（图片、语音）

## 注意事项

- 本系统需要Milvus向量数据库来存储和检索知识库向量
- 大模型配置需要根据实际情况修改，可选择本地部署或在线API
- 首次运行需要初始化数据库和知识库
- 生产环境需要修改SECRET_KEY等敏感配置
