# 人工智能 NLP-Agent 数字人项目 — 智能体任务 (MCP 版)

**工单编号**：人工智能NLP-Agent数字人项目-智能体任务  
**实现方案**：MCP 协议 + LLM Prompt 驱动的智能体引擎  

---

## 架构

```
用户输入 → Agent引擎(LLM决策) → MCP协议 → MCP工具服务器 → 各后端API → 结果返回
```

| 层 | 组件 | 作用 |
|:--:|------|------|
| 🧠 | `core/agent.py` | MCP客户端 + LLM Prompt意图识别/工具选择 |
| 🔌 | `mcp_server.py` | MCP服务器，将5个工具暴露为 MCP Tool |
| 🌐 | `frontend/web_app.py` | Flask Web 聊天界面 (port 6001) |
| ⚙️ | `config/config.yaml` | LLM API Key + 各工具地址 |

## 5 个后端服务

| 工具 | 项目路径 | 端口 | API |
|:----:|---------|:----:|:----:|
| 记账本 | `C:\Users\freedom\Desktop\agent\实训二工单一` | 8080 | POST `/api/chat` |
| 日程提醒 | `C:\Users\freedom\Desktop\实训二工单二` | 5000 | POST `/chat` |
| 文生图 | `C:\Users\freedom\Desktop\实训二工单三` | 7860 | POST `/generate` |
| 基金问答 | `C:\Users\freedom\Desktop\基金问答智能体` | 5002 | POST `/ask` |
| 招股书问答 | `C:\Users\freedom\Desktop\招股书问答智能体` | 5003 | POST `/ask` |

## 项目结构

```
C:\Users\freedom\Desktop\agent\06-Agent智能体项目\
├── mcp_server.py           # MCP 工具服务器
├── core/
│   └── agent.py            # Agent引擎（MCP客户端+LLM决策）
├── frontend/
│   └── web_app.py          # Flask Web 聊天界面
├── config/
│   └── config.yaml         # 配置（需填入API Key）
├── docs/
│   └── 开发文档.md           # 实现步骤与测试用例
├── requirements.txt
├── 启动所有服务.bat          # 一键启动全部
└── 启动CLI模式.bat           # 命令行测试模式
```

## 使用步骤

### 1️⃣ 配置 API Key
编辑 `config/config.yaml`，填入你的 LLM API Key：
```yaml
llm:
  api_key: "sk-xxxx"      # 你的 SiliconFlow / OpenAI Key
  base_url: "https://api.siliconflow.cn/v1"
  model: "Qwen/Qwen2.5-7B-Instruct"
```

### 2️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 3️⃣ 启动全部服务
双击 **`启动所有服务.bat`** 即可。

---

## MCP 优势

- **动态发现**: Agent自动发现工具，无需硬编码
- **标准协议**: 遵循 MCP 规范，可接入任意 MCP 兼容工具
- **可扩展**: 加一个新工具只需在 `mcp_server.py` 加一个 `@mcp.tool()`
