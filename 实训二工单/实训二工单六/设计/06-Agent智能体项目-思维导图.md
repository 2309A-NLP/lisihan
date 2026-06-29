# 06-Agent 智能体项目 — 思维导图

> 工单编号：人工智能NLP-Agent数字人项目-智能体任务
> 实现方案：MCP 协议 + LLM Prompt 驱动的智能体引擎

---

## 1. 项目架构总览

### 1.1 数据流
```
用户 → Flask Web UI (6001) → Agent引擎 (LLM决策) → MCP协议 → MCP工具服务器 → 各后端API → 结果返回
```

### 1.2 四层架构
- **🧠 智能决策层** — `core/agent.py` + `core/json_utils.py`
  - LLM Prompt 意图识别（SiliconFlow / OpenAI 兼容接口）
  - 关键词回退兜底
  - 对话历史管理（最近6轮）
  - MCP Client / HTTP 直连双模式
- **🔌 通信协议层** — MCP stdio 协议
  - `mcp_server.py` — FastMCP 服务器（5个工具暴露为 MCP Tool）
  - `core/mcp_client.py` — 自定义 MCP stdio 客户端（Content-Length 帧协议）
  - 支持 stdio（默认）和 HTTP SSE（--http 模式）双传输
- **🌐 前端交互层** — `frontend/web_app.py`
  - Flask Web 界面（端口 6001）
  - 工具侧边栏（记账/日程/文生图/基金/招股书）
  - 首页智能路由 + 工具视图直达 + 延续对话
  - 异步图片生成 + 图片编辑（人脸旋转/风格处理）
- **⚙️ 后端服务层** — 5个独立后端服务
  - 记账本（8081） | 日程提醒（5000） | 文生图（7860）
  - 基金问答（5002） | 招股书问答（5003）

---

## 2. 5 大 MCP 工具

### 2.1 📊 记账本 (ledger_query)
- **后端服务**：`实训二工单一` → `family_accounting_agent/app.py` → 端口 8081
- **API**：`POST /api/chat` | 参数：`message`
- **功能**：
  - 记录收入/支出（添加、修改、删除）
  - 按日期/类别/成员筛选
  - 月度/年度收支汇总、分类统计
  - 预算管理
- **MCP 工具名**：`ledger_query(query: str)`
- **超时**：30s | **重试**：2次
- **特殊**：支持多步延续对话（最多8步）

### 2.2 📅 日程提醒 (schedule_query)
- **后端服务**：`实训二工单二` → `daily_scheduler_agent/web_app.py` → 端口 5000
- **API**：`POST /chat` | 参数：`message`
- **功能**：
  - 添加/查询/修改/删除日程
  - 定时提醒设置
  - 按日期查看安排
- **MCP 工具名**：`schedule_query(query: str)`
- **超时**：30s | **重试**：2次

### 2.3 🎨 文生图 (image_generate)
- **后端服务**：`实训二工单三` → `webui.py` → 端口 7860
- **API**：`POST /generate`
- **功能**：
  - 文本生成图像（豆包 Doubao-Seedream-4.5 优先）
  - SiliconFlow FLUX.1-dev / SD3.5 降级
  - Pollinations.ai 终极兜底
  - 图片编辑（人脸旋转、Qwen-Image-Edit）
  - 异步生成（后台线程 + 轮询状态）
- **MCP 工具名**：`image_generate(prompt: str)`
- **超时**：120s | **重试**：1次

### 2.4 📈 基金数据问答 (fund_query)
- **后端服务**：`基金问答智能体` → `code/app.py` → 端口 5002
- **API**：`POST /ask` | 参数：`question`
- **功能**：
  - 按代码/名称查询基金信息
  - 最新净值、历史净值
  - 收益率统计、基金对比
  - 市场行情、板块热点
- **MCP 工具名**：`fund_query(question: str)`
- **超时**：30s | **重试**：2次

### 2.5 📄 招股说明书问答 (prospectus_query)
- **后端服务**：`招股书问答智能体` → `code/app.py` → 端口 5003
- **API**：`POST /ask` | 参数：`question`
- **功能**：
  - 财务数据（营收、利润、现金流）
  - 公司信息（背景、股东、管理层）
  - 募投项目详情、发行信息
  - 风险因素披露
- **MCP 工具名**：`prospectus_query(question: str)`
- **超时**：120s（RAG 慢查询）| **重试**：1次

---

## 3. 核心组件详解

### 3.1 Agent 引擎 (core/agent.py)
- **架构**：基于 MCP 协议 + LLM Prompt
- **工作流**：
  1. 问候语快速识别（跳过 LLM 调用）
  2. LLM 意图识别（8s 超时 → 关键词回退）
  3. 工具调用（MCP → HTTP 直连降级）
  4. 单工具直接返回 | 多工具 LLM 整合
- **关键特性**：
  - 安全兜底：LLM 将工具调用写成 direct_reply 时的强制修正
  - 持久化事件循环（process_sync 避免跨循环 session 冲突）
  - 对话历史管理（自动记录最近6轮）
  - 双通道：MCP stdio 或 HTTP 直连

### 3.2 MCP 服务器 (mcp_server.py)
- **框架**：FastMCP (mcp.server.fastmcp)
- **传输**：stdio（默认）| HTTP SSE（--http --port 8100）
- **配置**：5 个工具的后端地址、端口、超时、重试策略
- **降级策略**：指数退避重试（1s, 2s, 4s...）
- **健康检查**：`health://status` resource

### 3.3 MCP 客户端 (core/mcp_client.py)
- **协议**：JSON-RPC 2.0 + Content-Length 帧协议
- **连接**：
  - 启动 MCP 服务器子进程
  - 发送 initialize → 接收 initialized → tools/list
- **通信**：后台读取线程 + Queue 异步响应匹配
- **容错**：BrokenPipeError 检测 + 超时控制
- **参数适配**：自动转换 query→question（fund/prospectus）

### 3.4 JSON 提取工具 (core/json_utils.py)
- **四层策略**：
  1. 括号平衡提取 + 直接解析
  2. 常见错误修复（缺引号、尾逗号、缺括号）
  3. 正则手动提取（4种匹配模式）
  4. 关键词回退（最终保底）
- **修复类型**：缺闭合括号、多余逗号、重复布尔值、key 缺引号

### 3.5 Web UI (frontend/web_app.py)
- **框架**：Flask（端口 6001）
- **前端**：自建 HTML/CSS/JS 单页应用
  - 金色主题 UI，5个工具侧边栏
  - 欢迎页 + 快捷功能按钮
  - 聊天消息区 + 工具面板
- **路由**：
  - `/` — 主页面
  - `/chat` — 聊天处理（POST）
  - `/health` — 健康检查
  - `/tool_records` — 工具历史/实时数据
  - `/generate_image_async` — 异步文生图
  - `/image_status/<task_id>` — 图片生成状态
- **图片处理**：上传→豆包编辑→PIL 降级旋转→SiliconFlow
- **延续对话**：记账本多步流程（最多8步，完成信号自动重置）

### 3.6 启动器 (launcher.py)
- **功能**：一键启动 5 个后端服务 + Web UI
- **进程管理**：ProcessManager（启动、端口等待、状态检查、优雅停止）
- **Windows 特殊处理**：隐藏窗口、CREATE_NO_WINDOW
- **日志**：每个服务独立日志文件（按日期切割）

---

## 4. 配置文件

### 4.1 config/config.yaml
```yaml
llm:          # API Key, base_url, model, temperature, max_tokens
tools:        # 5个后端服务地址和端点
server:       # Web UI host + port
mcp:          # 传输模式 (stdio/sse) + HTTP 端口
```

### 4.2 启动脚本
- `启动所有服务.bat` → launcher.py
- `启动CLI模式.bat` → core/agent.py 命令行交互
- `install_startup.bat` — 安装开机自启
- `uninstall_startup.bat` — 卸载开机自启

### 4.3 依赖
```
flask>=3.0    requests>=2.31    pyyaml>=6.0    mcp>=1.2.0
```

---

## 5. 部署架构

### 5.1 组件拓扑
```
┌──────────────────────────────────────────────────┐
│  Windows 宿主机                                  │
│                                                    │
│  ┌─────────┐   ┌──────────┐   ┌────────────────┐ │
│  │  Web UI │──▶│  Agent   │──▶│  MCP Server    │ │
│  │ :6001   │   │  Engine  │   │  (stdio)       │ │
│  └─────────┘   └──────────┘   └───────┬────────┘ │
│                                        │           │
│  ┌────────┬────────┬────────┬─────────┴────────┐ │
│  │Ledger  │Schedule│ImgGen  │ FundQA │Prospectus│ │
│  │:8081   │:5000   │:7860   │ :5002  │ :5003    │ │
│  └────────┴────────┴────────┴────────┴──────────┘ │
│                                                    │
│  LLM API (SiliconFlow / 火山引擎) ◀── Internet     │
└──────────────────────────────────────────────────┘
```

### 5.2 端口分配
| 服务 | 端口 | 协议 |
|------|:----:|:----:|
| Agent Web UI | 6001 | HTTP |
| 记账本 | 8081 | HTTP |
| 日程提醒 | 5000 | HTTP |
| 文生图 | 7860 | HTTP |
| 基金问答 | 5002 | HTTP |
| 招股书问答 | 5003 | HTTP |
| MCP HTTP SSE (可选) | 8100 | HTTP |

---

## 6. 验收标准

### 6.1 工具选择准确率
- **总准确率**：≥ 90%
- **各工具准确率**：≥ 95%
- 测试工具：`benchmark_accuracy.py`

### 6.2 测试用例分布
- 记账本：7 个
- 日程提醒：6 个
- 文生图：6 个
- 基金问答：6 个
- 招股书：6 个
- 无需工具（问候/闲聊）：6 个
- 多工具协同：4 个
- 边界/歧义：3 个

---

> 生成日期：2026-06-29
