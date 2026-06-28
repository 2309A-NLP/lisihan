# 招股书问答智能体 — 部署报告

## 1 项目概述

| 项目 | 内容 |
|------|------|
| 项目名称 | 招股书问答智能体 |
| 所属赛事 | 博金杯金融数据挖掘赛 (RAG赛道) |
| 项目目标 | 基于80份招股说明书TXT文档构建智能问答系统，支持自然语言查询和复杂推理 |
| 技术架构 | 双架构：RAG管道 + ReAct Agent |
| 核心模型 | Qwen2.5-72B-Instruct (LLM)、BAAI/bge-large-zh-v1.5 (Embedding) |
| API提供商 | SiliconFlow (硅基流动) |
| 推理框架 | FAISS 向量检索 + Flask Web服务 |

## 2 环境依赖

### 2.1 硬件环境

| 资源 | 要求 |
|------|------|
| CPU | 4核+ (推荐8核) |
| 内存 | 16GB+ (推荐32GB) |
| 磁盘 | 5GB+ (索引221MB + 数据+代码) |
| 网络 | 需要访问 SiliconFlow API |

### 2.2 软件环境

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 | conda 虚拟环境 py310 |
| Flask | 2.x | Web服务框架 |
| faiss-cpu | 1.7.x | 向量检索库 |
| numpy | 1.24+ | 向量运算 |
| requests | 2.x | HTTP API调用 |
| OS | Windows 10/11 | 主机运行 (WSL也可) |

### 2.3 API密钥

```
API_KEY = "sk-xxx..."  SiliconFlow API Key
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
API_BASE = "https://api.siliconflow.cn/v1"
```

## 3 部署步骤

### 3.1 环境准备

```bash
# 创建 conda 环境
conda create -n py310 python=3.10
conda activate py310

# 安装依赖
pip install flask faiss-cpu numpy requests
```

### 3.2 项目文件结构

确保以下完整目录结构存在：

```
D:\招股书问答智能体\
├── code\
│   ├── app.py
│   ├── rag_core.py
│   ├── react_agent.py
│   ├── build_index_resilient.py
│   ├── batch_test.py
│   ├── classify_questions.py
│   ├── config.py (可选)
│   └── templates\index.html
├── data\
│   └── 招股书问题.jsonl
├── output\
│   ├── prospectus_index.faiss
│   └── prospectus_chunks.json
└── bs_challenge_financial_14b_dataset\
    ├── pdf_txt_file\ (80个TXT)
    └── dataset\博金杯比赛数据.db
```

### 3.3 索引构建

```bash
cd D:\招股书问答智能体\code
python build_index_resilient.py
```

- 读取 `pdf_txt_file/` 下所有TXT文件
- 按400字符分块（80字符重叠），共约221MB向量索引
- 使用 BGE-large-zh-v1.5 批量向量化（每批16条）
- 支持断点续跑：中断后自动从上次进度继续
- 输出：`output/prospectus_index.faiss` + `output/prospectus_chunks.json`

### 3.4 启动Web服务

方式一：双击 `启动Web服务.bat`

```bat
@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
C:\Users\freedom\.conda\envs\py310\python.exe code\app.py
pause
```

方式二：命令行

```bash
cd D:\招股书问答智能体
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python code\app.py
```

服务启动于 `http://localhost:5003`

### 3.5 批量测试

```bash
cd D:\招股书问答智能体
python code\batch_test.py
```

- 自动加载 `data/招股书问题.jsonl` (397题)
- 逐题调用 ReAct Agent 推理回答
- 每20题自动保存进度到 `output/prospectus_results.jsonl`
- 支持断点续跑：已有结果自动跳过

### 3.6 问题分类（可选）

```bash
python code\classify_questions.py
```

- 从原始 `question.json` 中筛选出招股书问题
- 输出：`data/招股书问题.jsonl`
- 分类规则：6层规则（强关键词 → 公司名白名单 → 基金关键词 → 正则匹配 → 兜底）

## 4 API接口

### 4.1 Web接口

```
POST http://localhost:5003/ask
Content-Type: application/json

{
    "question": "华瑞电器股份有限公司获得多少项国内专利？"
}
```

返回：

```json
{
    "answer": "根据招股说明书，...",
    "type": "AI",
    "error": ""
}
```

## 5 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CHUNK_SIZE | 400 | 每块字符数 |
| CHUNK_OVERLAP | 80 | 块重叠字符数 |
| BATCH_SIZE | 16 (索引) / 32 (查询) | 向量化批大小 |
| TOP_K | 5 | 检索返回片段数 |
| LLM_TEMPERATURE | 0.01 | LLM生成温度 |
| MAX_TOKENS | 1024/2000 | LLM最大输出 |
| MAX_RETRIES | 3-10 | API调用重试次数 |
| EMBEDDING_TIMEOUT | 60 | 向量化超时 |
| LLM_TIMEOUT | 60 | LLM调用超时 |
| SAVE_EVERY | 20 | 批量测试保存间隔 |
| REACT_MAX_STEPS | 10 | ReAct最大推理步数 |
| PORT | 5003 | Web服务端口 |

## 6 验证清单

| 检查项 | 方法 | 预期 |
|--------|------|------|
| 索引可用 | 确认文件存在 | prospectus_index.faiss ~221MB |
| Web服务 | 访问 localhost:5003 | 显示深色主题问答界面 |
| 单题问答 | 发POST /ask | 5秒内返回中文答案 |
| 批量测试 | 运行 batch_test.py | 逐题输出OK/FAIL |
| 问题分类 | 运行 classify_questions.py | 输出397题分类统计 |
| 中文兼容 | 输入含中文问题 | 正常返回不报编码错误 |
| 断点续跑 | 中断后重跑 | 跳过已有题继续处理 |
| 容错 | 断开网络再提问 | 返回知识兜底答案 |

## 7 常见问题

**Q: 索引构建失败**
A: 检查API密钥有效性、网络连通性。脚本支持断点续跑，重试即可。

**Q: Web服务启动报编码错误**
A: 使用启动脚本（已设 chcp 65001 + PYTHONUTF8=1），或在终端先执行 `chcp 65001`。

**Q: faiss读取中文路径失败**
A: react_agent.py 已实现 numpy.frombuffer 绕过方案，通过Python open()读取文件字节再反序列化。

**Q: API限流**
A: 索引构建使用小批量(16) + 指数退避重试(10次)；查询使用0.2-3秒间隔。

**Q: 答案质量不理想**
A: 可调整TOP_K、优化chunk策略、或增加搜索关键词多样性。

---

*部署完成时间：2026年6月28日*
