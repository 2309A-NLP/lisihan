# 06-Agent 智能体项目 — 部署文档

> 工单编号：人工智能NLP-Agent数字人项目-智能体任务
> 项目路径：`C:\Users\freedom\Desktop\agent\06-Agent智能体项目`
> Python 环境：Conda py310 (`C:\Users\freedom\.conda\envs\py310`)

---

## 目录

1. [系统要求](#1-系统要求)
2. [环境准备](#2-环境准备)
3. [依赖安装](#3-依赖安装)
4. [配置文件](#4-配置文件)
5. [启动后端服务](#5-启动后端服务)
6. [启动 Agent 系统](#6-启动-agent-系统)
7. [访问与使用](#7-访问与使用)
8. [项目管理](#8-项目管理)
9. [故障排除](#9-故障排除)
10. [卸载](#10-卸载)

---

## 1. 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（本部署文档以 Windows 为准） |
| Python | 3.10+（推荐 Conda py310 环境） |
| 网络 | 访问 SiliconFlow / 火山引擎 API（文生图） |
| 内存 | ≥ 8GB（招股书 RAG 服务可能需要 4GB+） |
| 磁盘 | ≥ 1GB（含日志和静态文件） |

### 项目依赖总览

| 依赖 | 版本 | 用途 |
|------|------|------|
| flask | ≥3.0 | Web 界面框架 |
| requests | ≥2.31 | HTTP 调用后端服务 |
| pyyaml | ≥6.0 | 配置文件解析 |
| mcp | ≥1.2.0 | MCP 协议支持 |

---

## 2. 环境准备

### 2.1 确认 Python 环境

```batch
:: 查看 Conda 环境
conda env list

:: 激活项目使用的 py310 环境
call C:\Users\freedom\.conda\envs\py310\Scripts\activate.bat
python --version
:: 应输出: Python 3.10.x
```

### 2.2 项目结构校验

部署前确认以下文件/目录存在：

```
C:\Users\freedom\Desktop\agent\06-Agent智能体项目\
├── mcp_server.py            ✅ MCP 工具服务器
├── core/
│   ├── agent.py             ✅ Agent 引擎
│   ├── json_utils.py        ✅ JSON 提取工具
│   └── mcp_client.py        ✅ MCP 客户端
├── frontend/
│   ├── web_app.py           ✅ Flask Web 界面
│   └── templates/
│       └── index.html       ✅ 前端页面
├── config/
│   └── config.yaml          ✅ 配置文件
├── launcher.py              ✅ 一键启动器
├── requirements.txt         ✅ 依赖清单
├── 启动所有服务.bat          ✅ 快捷启动
└── 启动CLI模式.bat           ✅ CLI 模式启动
```

### 2.3 确认 5 个后端服务目录

| 工具 | 期望路径 | 确认 |
|------|---------|:----:|
| 记账本 | `C:\Users\freedom\Desktop\agent\实训二工单一\family_accounting_agent\app.py` | ☐ |
| 日程提醒 | `C:\Users\freedom\Desktop\agent\实训二工单二\daily_scheduler_agent\web_app.py` | ☐ |
| 文生图 | `C:\Users\freedom\Desktop\agent\实训二工单三\webui.py` | ☐ |
| 基金问答 | `C:\Users\freedom\Desktop\agent\基金问答智能体\code\app.py` | ☐ |
| 招股书问答 | `C:\Users\freedom\Desktop\agent\招股书问答智能体\code\app.py` | ☐ |

> **注意**：路径在 `launcher.py` 中定义（第 53-58 行）。如果后端目录位置不同，需修改这里的路径。

---

## 3. 依赖安装

### 3.1 安装核心依赖

```batch
cd /d C:\Users\freedom\Desktop\agent\06-Agent智能体项目
C:\Users\freedom\.conda\envs\py310\python.exe -m pip install -r requirements.txt
```

### 3.2 验证安装

```batch
C:\Users\freedom\.conda\envs\py310\python.exe -c "import flask, requests, yaml, mcp; print('所有依赖已安装')"
```

---

## 4. 配置文件

### 4.1 配置 LLM API Key

编辑 `config/config.yaml`：

```yaml
llm:
  api_key: "sk-你的SiliconFlow_API_Key"    # ← 必填：替换为实际 Key
  base_url: "https://api.siliconflow.cn/v1"  # 或 OpenAI 兼容接口
  model: "Qwen/Qwen2.5-7B-Instruct"          # 推荐：通义千问 7B
  temperature: 0.1                            # 低温度 = 更确定
  max_tokens: 1024                            # 最大输出 Token
```

> 💡 **API Key 获取**：
> - SiliconFlow：https://siliconflow.cn → 控制台 → API Key
> - 更稳定的文生图也可以使用火山引擎（豆包 Doubao-Seedream-4.5），Key 已内置于 `web_app.py` 第 113-115 行

### 4.2 工具地址配置（一般无需修改）

```yaml
tools:
  ledger:
    base_url: "http://127.0.0.1:8081"
    chat_endpoint: "/api/chat"
  schedule:
    base_url: "http://127.0.0.1:5000"
    chat_endpoint: "/chat"
  image_gen:
    base_url: "http://127.0.0.1:7860"
    generate_endpoint: "/generate"
  fund:
    base_url: "http://127.0.0.1:5002"
    chat_endpoint: "/ask"
  prospectus:
    base_url: "http://127.0.0.1:5003"
    chat_endpoint: "/ask"
```

> ⚠️ **端口冲突处理**：如果某个端口已被占用，修改该工具的 `base_url` 端口号，并确保对应后端服务监听在新端口上。

### 4.3 服务器配置

```yaml
server:
  host: "127.0.0.1"       # 监听地址（生产环境改为 0.0.0.0 需加防火墙）
  port: 6001              # Web UI 端口
```

---

## 5. 启动后端服务

有两种方式启动后端 5 个服务：

### 方式 A：使用启动器一键启动（推荐）

```batch
cd /d C:\Users\freedom\Desktop\agent\06-Agent智能体项目
C:\Users\freedom\.conda\envs\py310\python.exe launcher.py
```

或双击 `启动所有服务.bat`。

启动器会自动：
1. 逐一启动 5 个后端服务
2. 等待每个服务就绪（端口监听检测）
3. 显示服务状态报告
4. 启动 Agent Web UI（端口 6001）
5. 自动打开浏览器

**预期输出示例**：
```
[Launcher] ============================================================
[Launcher] 启动服务...
[Launcher] ============================================================
[Launcher] [Ledger] 启动中 (端口 8081)...
[Launcher] [Ledger] 等待 8s 让服务初始化...
[Launcher] [Schedule] 启动中 (端口 5000)...
...
[Launcher] ✅ 所有服务已启动
[Launcher]    Web UI: http://127.0.0.1:6001
[Launcher]    日志目录: C:\...\06-Agent智能体项目\logs
[Launcher]    按 Ctrl+C 停止所有服务
```

### 方式 B：逐一手动启动（调试用）

```batch
:: 启动记账本
start "" "C:\Users\freedom\.conda\envs\py310\python.exe" "C:\Users\freedom\Desktop\agent\实训二工单一\family_accounting_agent\app.py"

:: 启动日程提醒
start "" "C:\Users\freedom\.conda\envs\py310\python.exe" "C:\Users\freedom\Desktop\agent\实训二工单二\daily_scheduler_agent\web_app.py"

:: 启动文生图
start "" "C:\Users\freedom\.conda\envs\py310\python.exe" "C:\Users\freedom\Desktop\agent\实训二工单三\webui.py"

:: 启动基金问答
start "" "C:\Users\freedom\.conda\envs\py310\python.exe" "C:\Users\freedom\Desktop\agent\基金问答智能体\code\app.py"

:: 启动招股书问答
start "" "C:\Users\freedom\.conda\envs\py310\python.exe" "C:\Users\freedom\Desktop\agent\招股书问答智能体\code\app.py"
```

---

## 6. 启动 Agent 系统

### 6.1 Web 模式（推荐）

后端服务就绪后，启动 Agent Web UI：

```batch
cd /d C:\Users\freedom\Desktop\agent\06-Agent智能体项目
C:\Users\freedom\.conda\envs\py310\python.exe frontend\web_app.py
```

浏览器打开 `http://127.0.0.1:6001`

### 6.2 CLI 模式（测试用）

```batch
cd /d C:\Users\freedom\Desktop\agent\06-Agent智能体项目
C:\Users\freedom\.conda\envs\py310\python.exe core\agent.py
```

或双击 `启动CLI模式.bat`。

CLI 模式会连接到 MCP 服务器并启动命令行交互界面：
```
==================================================
  智能管家 Agent v2.0 (MCP)
  工单编号：人工智能NLP-Agent数字人项目-智能体任务
==================================================

[MCP] Ready, 5 tools found

  [tool] ledger_query: 📊 记账本 — 管理个人收支记录...
  [tool] schedule_query: 📅 日程提醒 — 管理日程事件...
  [tool] image_generate: 🎨 文生图 — 根据文本描述生成...
  [tool] fund_query: 📈 基金数据问答 — 查询基金实时数据...
  [tool] prospectus_query: 📄 招股说明书问答 — 解析招股说明书...

>>> 记录今天午餐花了35元
```

---

## 7. 访问与使用

### 7.1 Web UI 访问

| 环境 | 地址 |
|------|------|
| 本机 | `http://127.0.0.1:6001` |
| 局域网 | `http://你的IP:6001`（需将 host 改为 `0.0.0.0`） |

### 7.2 健康检查

```batch
curl http://127.0.0.1:6001/health
```

预期响应：
```json
{
  "status": "ok",
  "mcp": true,
  "agent_ready": true,
  "agent_mode": "MCP",
  "mcp_tools": ["ledger_query", "schedule_query", "image_generate", "fund_query", "prospectus_query"]
}
```

### 7.3 工具测试命令

| 测试内容 | 输入示例 |
|---------|----------|
| 记账本 | "记录今天午餐花了35元" |
| 日程提醒 | "提醒我明天下午3点开会" |
| 文生图 | "画一只可爱的橘猫" |
| 基金查询 | "查一下000001基金的最新净值" |
| 招股书查询 | "查一下这家公司的营收情况" |
| 多工具 | "查我花了多少钱，再看看000001基金的净值" |
| 闲聊 | "你好" / "你是谁" |

---

## 8. 项目管理

### 8.1 服务状态检查

```batch
:: 检查端口监听
netstat -ano | findstr ":6001"
netstat -ano | findstr ":8081"
netstat -ano | findstr ":5000"
netstat -ano | findstr ":5002"
netstat -ano | findstr ":5003"

:: 查看所有 Agent 相关进程
tasklist | findstr "python"
```

### 8.2 日志查看

所有日志位于 `logs/` 目录，按日期切割：

| 日志文件 | 内容 |
|---------|------|
| `launcher_YYYYMMDD.log` | 启动器日志 |
| `ui_YYYYMMDD.log` | Web UI 日志 |
| `ledger_YYYYMMDD.log` | 记账本服务日志 |
| `schedule_YYYYMMDD.log` | 日程提醒服务日志 |
| `imagegen_YYYYMMDD.log` | 文生图服务日志 |
| `fundqa_YYYYMMDD.log` | 基金问答服务日志 |
| `prospectus_YYYYMMDD.log` | 招股书问答服务日志 |
| `agent_ui_test.log` | Agent 测试日志 |

### 8.3 指定端口启动

```batch
:: 使用自定义端口
python launcher.py --port 6002 --no-browser
```

### 8.4 停止所有服务

按 `Ctrl+C` 即可优雅停止所有服务。

---

## 9. 故障排除

### 9.1 通用排查流程

```
症状 → 1. 检查端口是否监听
       2. 查看对应日志文件
       3. 检查 API Key 是否配置
       4. 确认网络可达
```

### 9.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **🚫 Web UI 无法访问** | Flask 未启动 / 端口被占用 | 检查 6001 端口：`netstat -ano \| findstr :6001` |
| **🔴 MCP 连接失败** | MCP 服务器脚本路径不对 / Python 版本不匹配 | 检查 `mcp_server.py` 是否存在，确保 Conda 环境正确 |
| **⚠️ 工具返回"服务暂不可用"** | 对应后端服务未启动 | 检查该工具端口是否监听 |
| **❌ 文生图失败** | 豆包 Key 过期 / SiliconFlow 余额不足 | 查看 `imagegen_*.log` 中的具体错误 |
| **🐢 招股书查询慢** | RAG 搜索耗时（正常 30-90s） | 超时已设为 120s，请耐心等待 |
| **🔑 API Key 错误** | Key 为空或示例值 | 编辑 `config.yaml` 填入真实 Key |
| **🔄 记账本确认循环** | 延续对话超过 8 步限制 | 自动终止，可重新输入完整指令 |
| **📝 LLM 决策不准** | Prompt 未覆盖某些场景 | 可修改 `agent.py` 中 `PROMPT_TEMPLATE` 补充规则 |

### 9.3 端口被占用处理

```batch
:: 查找占用端口的进程
netstat -ano | findstr :6001
:: 输出最后一列为 PID

:: 终止进程
taskkill /PID <PID> /F
```

### 9.4 MCP 日志调试

```batch
:: 设置 MCP 日志级别为 DEBUG
set MCP_LOG_LEVEL=DEBUG
python mcp_server.py --http --port 8100
```

---

## 10. 卸载

### 10.1 停止服务

按 `Ctrl+C` 停止启动器，或手动结束所有 Python 进程：

```batch
taskkill /F /IM python.exe
```

### 10.2 删除项目

```batch
rmdir /S /Q C:\Users\freedom\Desktop\agent\06-Agent智能体项目
```

### 10.3（可选）卸载开机自启

如果有安装过开机自启，双击 `uninstall_startup.bat` 即可移除。

---

> **部署验证清单**
>
> - [ ] Conda py310 环境可用
> - [ ] 所有依赖已安装
> - [ ] config.yaml 中 API Key 已配置
> - [ ] 5 个后端服务目录存在（路径匹配 launcher.py）
> - [ ] 5 个后端服务端口未被占用
> - [ ] 能通过 launcher 一键启动
> - [ ] Web UI http://127.0.0.1:6001 可访问
> - [ ] 至少一个工具调用成功
>
> 生成日期：2026-06-29
