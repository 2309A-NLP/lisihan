"""
Smart Agent Web UI — MCP stdio (custom client)
工单编号：人工智能NLP-Agent数字人项目-智能体任务

功能：
1. 基于 Flask 的 Web 界面
2. 集成 MCP 客户端和 Agent 引擎
3. 支持多工具（记账、日程、文生图、基金、招股书）
4. 异步图片生成（资源密集型任务）
5. 对话历史和状态管理（所有工具统一记录）
6. 支持工具视图延续对话（记账本多步流程）
"""

import os
import sys
import threading
import uuid
import json as _json
import base64 as _b64
import io
import time
import urllib.parse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List, Any

# ===== 路径设置 =====
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

# ===== Flask 导入 =====
from flask import Flask, jsonify, render_template, request

# ===== 项目模块导入 =====
from core.agent import Agent
from core.mcp_client import McpClient

# ===== 第三方库导入 =====
import requests as http_requests
import yaml

# ===== 应用初始化 =====
# 获取当前文件所在目录，用于定位模板文件夹
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(FRONTEND_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ===== 日志配置 =====
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("WebApp")

# ============================================================
# 1. MCP 客户端 (后台线程连接，带重试机制)
# ============================================================

PYTHON_EXE = r"C:\Users\freedom\.conda\envs\py310\python.exe"
MCP_SCRIPT = os.path.join(project_dir, "mcp_server.py")

# 创建 MCP 客户端实例
mcp = McpClient(PYTHON_EXE, MCP_SCRIPT)
mcp_connected = False


def _connect_mcp():
    """
    后台线程：连接 MCP 服务器，最多尝试 3 次
    """
    global mcp_connected
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"[MCP] 连接尝试 {attempt + 1}/{max_attempts}...")
            ok = mcp.connect(timeout=30)
            if ok:
                mcp_connected = True
                tools = [t["name"] for t in mcp.tools]
                logger.info(f"[MCP] 已连接, {len(tools)} tools: {tools}")
                return
            else:
                logger.warning(f"[MCP] 连接尝试 {attempt + 1} 失败")
                if attempt < max_attempts - 1:
                    time.sleep(3)
        except Exception as e:
            logger.error(f"[MCP] 连接异常 (尝试 {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(3)
    logger.warning("[MCP] 所有连接尝试失败 (将使用关键词回退)")


t = threading.Thread(target=_connect_mcp, daemon=True)
t.start()

# ============================================================
# 2. LLM 配置加载
# ============================================================

config_path = os.path.join(project_dir, "config", "config.yaml")
LLM_CONFIG = {}
if os.path.exists(config_path):
    with open(config_path, encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}
        LLM_CONFIG = full_config.get("llm", {})
        logger.info(f"[Config] LLM 配置加载完成: model={LLM_CONFIG.get('model', 'N/A')}")
else:
    logger.warning(f"[Config] 配置文件不存在: {config_path}")

# ===== 豆包（火山引擎）文生图配置 =====
DOUBAO_API_KEY = "ark-7419c003-7697-4d54-bbec-f08ad08094e5-25ec6"
DOUBAO_ENDPOINT = "ep-20260625141102-6t8th"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_IMAGE_SIZE = "1920x1920"

# ============================================================
# 3. Agent 引擎
# ============================================================

_agent_engine = Agent(MCP_SCRIPT, LLM_CONFIG)
_agent_ready = False


def _ensure_agent_connected():
    """
    预连接 Agent 引擎，即使 MCP 失败也可用 HTTP 直连
    """
    global _agent_ready
    if not _agent_ready:
        try:
            _agent_engine.sync_connect()
            tool_count = len(_agent_engine._tools_cache)
            _agent_ready = True
            logger.info(f"[Agent] 引擎就绪，{tool_count} 个工具可用")
        except Exception as e:
            logger.warning(f"[Agent] MCP连接可选（HTTP直连可用）: {e}")
            _agent_ready = True

# ============================================================
# 4. 工具配置
# ============================================================

# 所有工具名称映射
VALID_TOOLS = {
    "ledger_query": "记账本",
    "schedule_query": "日程提醒",
    "image_generate": "文生图",
    "fund_query": "基金问答",
    "prospectus_query": "招股书查询"
}

# 工具对应的 HTTP API 端点（MCP 失败时的降级方案）
TOOL_APIS = {
    "ledger_query": {"url": "http://127.0.0.1:8081/api/chat", "param": "message"},
    "schedule_query": {"url": "http://127.0.0.1:5000/chat", "param": "message"},
    "fund_query": {"url": "http://127.0.0.1:5002/ask", "param": "question"},
    "prospectus_query": {"url": "http://127.0.0.1:5003/ask", "param": "question"},
}

# ============================================================
# 5. 对话历史管理（统一记录所有工具的历史，持久化到文件）
# ============================================================

# 历史记录文件路径
HISTORY_DIR = os.path.join(project_dir, "data")
os.makedirs(HISTORY_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(HISTORY_DIR, "tool_history.json")

_tool_history: Dict[str, list] = {t: [] for t in VALID_TOOLS}

# 启动时从文件加载历史记录
def _load_history():
    """从 JSON 文件加载历史记录"""
    global _tool_history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = _json.load(f)
            # 只加载已知工具的历史
            for tool in VALID_TOOLS:
                if tool in data and isinstance(data[tool], list):
                    _tool_history[tool] = data[tool][-50:]  # 限 50 条
            logger.info(f"[History] 已加载历史记录 ({sum(len(v) for v in _tool_history.values())} 条)")
        except Exception as e:
            logger.warning(f"[History] 加载历史记录失败: {e}")

def _save_history():
    """保存历史记录到 JSON 文件"""
    try:
        # 只保存非空历史，减少写磁盘次数
        data = {k: v for k, v in _tool_history.items() if v}
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[History] 保存历史记录失败: {e}")

# 启动时加载
_load_history()

# 记录上一次使用的工具（用于延续对话）
_last_tool_view: Optional[str] = None


def _record_history(tool_name: str, user_input: str, reply: str):
    """
    记录对话历史到对应工具
    所有工具调用都会经过此函数，确保历史记录完整统一

    Args:
        tool_name: 工具名称（如 ledger_query）
        user_input: 用户输入
        reply: 回复内容
    """
    if tool_name in _tool_history:
        logger.info(f"[History] 记录 {tool_name}: {user_input[:30]}...")
        _tool_history[tool_name].append({
            "user": user_input[:200],
            "reply": reply[:500] if reply else "",
            "time": datetime.now().strftime("%H:%M")
        })
        # 只保留最近 50 条
        _tool_history[tool_name] = _tool_history[tool_name][-50:]
        # 持久化到文件
        _save_history()
    else:
        logger.warning(f"[History] 未知工具: {tool_name}")


# ============================================================
# 6. 工具执行函数（统一入口，自动记录历史）
# ============================================================

def _execute_tool(tool_name: str, message: str) -> str:
    """
    执行工具：优先使用 MCP，失败时降级到 HTTP 直连
    所有返回结果都会自动记录到历史

    Args:
        tool_name: 工具名称
        message: 用户输入

    Returns:
        回复内容
    """
    # ===== 文生图特殊处理 =====
    if tool_name == "image_generate":
        result = _call_siliconflow_txt2img(message)
        reply = result.get("reply", str(result)) if isinstance(result, dict) else str(result)
        _record_history(tool_name, message, reply)
        return reply

    # ===== MCP 调用 =====
    if mcp_connected:
        arg_map = {
            "ledger_query": "query",
            "schedule_query": "query",
            "fund_query": "question",
            "prospectus_query": "question"
        }
        arg_key = arg_map.get(tool_name, "query")
        try:
            with ThreadPoolExecutor() as ex:
                future = ex.submit(mcp.call_tool, tool_name, {arg_key: message}, 120)
                result = future.result(timeout=120)
                # 检查是否以"错误:"开头（中文错误检测）
                if result and not result.startswith("错误:"):
                    _record_history(tool_name, message, result)
                    return result
        except Exception as e:
            logger.warning(f"[MCP] 工具 {tool_name} 调用失败，降级到 HTTP: {e}")

    # ===== HTTP 直连降级 =====
    endpoint = TOOL_APIS.get(tool_name)
    if not endpoint:
        reply = f"未知工具: {tool_name}"
        _record_history(tool_name, message, reply)
        return reply

    try:
        payload = {endpoint["param"]: message}
        response = http_requests.post(endpoint["url"], json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        reply = result.get("reply") or result.get("answer") or result.get("text") or str(result)
        # 基金问答特殊处理：使用 answer 字段
        if tool_name == "fund_query" and result.get("answer"):
            reply = result.get("answer")
        _record_history(tool_name, message, reply)
    except http_requests.ConnectionError:
        reply = f"【{tool_name}】后端服务未启动，请先启动 {endpoint['url']}"
        _record_history(tool_name, message, reply)
        return reply
    except Exception as e:
        logger.error(f"[HTTP] 工具 {tool_name} 调用失败: {e}")
        reply = f"工具调用失败: {e}"
        _record_history(tool_name, message, reply)
        return reply


def _call_siliconflow_txt2img_full(prompt: str) -> dict:
    """
    生成图片：优先使用豆包（火山引擎），失败时降级
    """
    # ===== 1. 豆包（火山引擎 Doubao-Seedream-4.5）=====
    try:
        logger.info(f"[Image] 调用豆包: {prompt[:30]}...")
        headers = {"Authorization": f"Bearer {DOUBAO_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": DOUBAO_ENDPOINT,
            "prompt": prompt,
            "size": DOUBAO_IMAGE_SIZE,
            "n": 1,
        }
        response = http_requests.post(
            f"{DOUBAO_BASE_URL}/images/generations",
            headers=headers, json=payload, timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            img_url = data.get("data", [{}])[0].get("url", "")
            if img_url:
                img_resp = http_requests.get(img_url, timeout=30)
                if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                    img_b64 = _b64.b64encode(img_resp.content).decode()
                    # 保存到本地
                    try:
                        img_dir = os.path.join(project_dir, "static", "generated")
                        os.makedirs(img_dir, exist_ok=True)
                        filename = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        with open(os.path.join(img_dir, filename), "wb") as f:
                            f.write(img_resp.content)
                        logger.info(f"[Image] 图片已保存: {filename}")
                    except Exception as e:
                        logger.warning(f"[Image] 保存失败: {e}")
                    return {"reply": "图片已生成", "image": img_b64}
        logger.warning(f"[Image] 豆包返回: HTTP {response.status_code}")
    except Exception as e:
        logger.warning(f"[Image] 豆包调用失败: {e}")

    # ===== 2. 降级到 SiliconFlow =====
    return _siliconflow_fallback(prompt)


def _siliconflow_fallback(prompt: str) -> dict:
    """SiliconFlow 降级方案"""
    api_key = LLM_CONFIG.get("api_key", "")
    base_url = LLM_CONFIG.get("base_url", "https://api.siliconflow.cn/v1")
    img_config_path = os.path.join(project_dir, "config", "config.yaml")
    if os.path.exists(img_config_path):
        with open(img_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            api_cfg = cfg.get("api", {})
            if api_cfg.get("api_key"):
                api_key = api_cfg["api_key"]
            if api_cfg.get("base_url"):
                base_url = api_cfg["base_url"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    IMAGE_MODELS = ["black-forest-labs/FLUX.1-dev", "stabilityai/stable-diffusion-3-5-large"]
    last_error = ""
    for model in IMAGE_MODELS:
        payload = {"model": model, "prompt": prompt, "image_size": "1024x1024", "num_inference_steps": 4}
        try:
            resp = http_requests.post(f"{base_url}/image/generations", headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                last_error = f"{model}: HTTP {resp.status_code}"; continue
            data = resp.json()
            for img in data.get("images", []):
                url = img.get("url", "")
                if url:
                    ir = http_requests.get(url, timeout=30)
                    if ir.status_code == 200 and len(ir.content) > 1000:
                        b64 = _b64.b64encode(ir.content).decode(); return {"reply": "图片已生成", "image": b64}
        except Exception as e:
            last_error = f"{model}: {str(e)[:60]}"; continue
    # Pollinations.ai 终极兜底
    try:
        logger.info("[Image] 降级到 Pollinations.ai")
        pr = http_requests.get(f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=1024&nologo=true", timeout=60)
        if pr.status_code == 200 and len(pr.content) > 1000:
            return {"reply": "图片已生成", "image": _b64.b64encode(pr.content).decode()}
    except:
        pass
    return {"reply": f"图片生成失败: {last_error}"}


def _call_siliconflow_txt2img(prompt: str) -> str:
    """兼容旧接口：返回字符串"""
    result = _call_siliconflow_txt2img_full(prompt)
    return result.get("reply", str(result))


def _keyword_decide_tool(message: str) -> str:
    """
    基于关键词的工具路由（回退方案）
    """
    msg_lower = message.lower()
    patterns = [
        ("ledger_query", ["记账", "记录", "花了", "支出", "收入", "账目", "账单", "消费", "流水"]),
        ("schedule_query", ["日程", "提醒", "会议", "安排", "日历"]),
        ("image_generate", ["画一幅", "画一张", "帮我画", "生成图片", "生成一张", "画图", "作图", "文生图", "绘制"]),
        ("fund_query", ["基金", "净值", "收益率", "涨幅"]),
        ("prospectus_query", ["招股", "说明书", "营收", "财务数据", "配售", "募投", "招股书"]),
    ]
    for tool_id, keywords in patterns:
        if any(kw in msg_lower for kw in keywords):
            return tool_id
    return "UNKNOWN"


# ============================================================
# 7. 判断是否为延续性回复
# ============================================================

# 记账本多步流程的延续关键词
# 注意："确认"虽然需要第一次延续，但需要防重复循环
CONTINUATION_WORDS = {
    "是", "取消",
    "今天", "昨天", "前天", "我", "妈妈", "爸爸", "女儿", "儿子",
}

# 只有记账本才支持多步延续对话，其他工具每次查询独立路由
CONTINUATION_TOOLS = {"ledger_query"}

# 最大延续步数，防止确认循环（记账本复杂流程可能需要5-8步）
MAX_CONTINUATION_STEPS = 8

# 当前延续步数
_continuation_step = 0

# 工具回复中的完成信号词（表示流程已结束，不需要再延续）
COMPLETION_SIGNALS = [
    "已记录", "已添加", "已更新", "已删除", "已完成", "已成功",
    "记录成功", "添加成功", "更新成功", "删除成功", "操作成功",
    "已保存", "保存成功", "已取消", "取消成功",
    "已生成", "图片已生成",
]


def _is_continuation(message: str) -> bool:
    """
    判断当前输入是否为延续性回复（仅记账本多步流程）

    防循环机制：
    1. 超过 MAX_CONTINUATION_STEPS 次后自动终止延续
    2. 检测到完成信号词（"已记录""已成功"等）后自动重置
    3. **短消息防误判**：即使 < 10 个字，但如果明确包含其他工具关键词
       （如"画""生成图片"等），不应作为延续路由

    Args:
        message: 用户输入

    Returns:
        True 表示继续延续到当前工具，False 表示终止延续
    """
    global _last_tool_view, _continuation_step
    # 只有记账本才支持延续对话
    if _last_tool_view not in CONTINUATION_TOOLS:
        _last_tool_view = None
        _continuation_step = 0
        return False
    # 超过最大步数限制，终止延续
    if _continuation_step >= MAX_CONTINUATION_STEPS:
        logger.info(f"[Continuation] 达到最大延续步数 ({MAX_CONTINUATION_STEPS})，终止延续")
        _last_tool_view = None
        _continuation_step = 0
        return False
    msg_lower = message.lower()

    # ===== 防误判：短消息如果包含其他工具关键词，不应作为延续 =====
    # 例如："画一条鱼"虽然是短消息，但明显是文生图请求
    non_continuation_triggers = [
        "画",       # 文生图
        "生成",     # 文生图
        "绘制",     # 文生图
        "基金",     # 基金查询
        "净值",     # 基金查询
        "招股",     # 招股书
        "提醒",     # 日程
        "会议",     # 日程
        "日程",     # 日程
        "图片",     # 文生图
    ]
    if any(kw in msg_lower for kw in non_continuation_triggers):
        # 但不完全放行 — 如果消息本来就是"今天明天"这类纯日期/延续词，并且包含延续关键词，仍作为延续
        if any(kw in msg_lower for kw in CONTINUATION_WORDS) and len(message) <= 4:
            pass  # 极短的延续词如"画"? "是"? 不, "画"应走文生图
        else:
            logger.info(f"[Continuation] 短消息含其他工具关键词，不延续: {message}")
            _last_tool_view = None
            _continuation_step = 0
            return False

    # 短消息（少于10个字）或者包含延续关键词
    if len(message) < 10:
        return True
    if any(kw in msg_lower for kw in CONTINUATION_WORDS):
        return True
    return False


def _check_completion(reply: str) -> bool:
    """
    检查工具回复中是否包含完成信号词。
    如果完成，自动重置延续状态，防止反复确认。

    Args:
        reply: 工具回复文本

    Returns:
        True 如果包含完成信号
    """
    global _last_tool_view, _continuation_step
    for signal in COMPLETION_SIGNALS:
        if signal in reply:
            logger.info(f"[Continuation] 检测到完成信号 '{signal}'，重置延续状态")
            _last_tool_view = None
            _continuation_step = 0
            return True
    return False


# ============================================================
# 8. Flask 路由
# ============================================================

@app.route("/")
def index():
    """主页：渲染前端页面"""
    return render_template('index.html')


@app.route("/health")
def health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "mcp": mcp_connected,
        "agent_ready": _agent_ready,
        "agent_mode": "HTTP直连" if _agent_ready and not _agent_engine._tools_cache else "MCP",
        "mcp_tools": [t["name"] for t in mcp.tools] if mcp_connected else [],
    })


@app.route("/tool_records")
def tool_records():
    """
    获取工具的历史记录或实时数据

    支持：
    - ledger: 从记账本后端获取真实数据（包含收入和支出汇总）
    - 其他: 从内存历史中获取
    """
    view = request.args.get("view", "home")

    tool_map = {
        "ledger": "ledger_query",
        "schedule": "schedule_query",
        "image": "image_generate",
        "fund": "fund_query",
        "prospectus": "prospectus_query"
    }

    tool_name = tool_map.get(view)
    if not tool_name:
        return jsonify({"type": "empty", "items": []})

    # ===== 记账本特殊处理：获取实时数据 =====
    if view == "ledger":
        try:
            # 获取记录列表
            resp = http_requests.get(
                "http://127.0.0.1:8081/api/records?limit=50",
                timeout=10
            )
            # 获取汇总统计
            summary_resp = http_requests.get(
                "http://127.0.0.1:8081/api/summary",
                timeout=10
            )

            records = []
            summary = None

            if resp.ok:
                data = resp.json()
                # 兼容两种返回格式：直接返回 records 或嵌套在 summary 中
                records = data.get("records", [])
                if not records and isinstance(data, dict):
                    records = data.get("summary", {}).get("recent", [])

            if summary_resp.ok:
                s = summary_resp.json()
                # 修正字段名：记账本返回的是 month_income 和 month_total
                summary = {
                    "total_income": s.get("month_income", 0),
                    "total_expense": s.get("month_total", 0),
                    "month": s.get("month", ""),
                }

            return jsonify({
                "type": "ledger",
                "records": records,
                "summary": summary,
            })
        except Exception as e:
            logger.warning(f"[Records] 获取记账数据失败: {e}")
            return jsonify({"type": "error", "message": f"记账本后端未启动 (8081)"})

    # ===== 其他工具：返回对话历史 =====
    history = _tool_history.get(tool_name, [])
    items = [{"time": h["time"], "text": h["user"][:80]} for h in history[-30:]]
    return jsonify({"type": "history", "items": items})


@app.route("/chat", methods=["POST"])
def chat():
    """
    聊天处理接口

    支持：
    1. JSON 请求（普通文本）
    2. FormData 请求（带图片上传）
    所有工具调用都会自动记录历史
    支持工具视图的延续对话（记账本多步流程）
    """
    global _last_tool_view, _continuation_step

    # ===== 解析请求 =====
    if request.content_type and 'multipart/form-data' in request.content_type:
        data = request.form.to_dict()
        data["image"] = request.form.get("image_b64", "")
    else:
        data = request.get_json(silent=True) or {}

    message = (data.get("message") or "").strip()
    view = (data.get("view") or "home").strip().lower()
    image_b64 = data.get("image")

    if not message and not image_b64:
        return jsonify({"reply": "请输入您的需求"})

    try:
        # ============================================================
        # 1. 图片上传处理
        # ============================================================
        if image_b64:
            return _handle_image_upload(image_b64, message, view)

        # ============================================================
        # 2. 文本消息处理
        # ============================================================

        # 2.1 如果指定了工具视图，直接调用对应工具
        view_to_tool = {
            "ledger": "ledger_query",
            "schedule": "schedule_query",
            "image": "image_generate",
            "fund": "fund_query",
            "prospectus": "prospectus_query"
        }

        forced_tool = view_to_tool.get(view)

        if forced_tool:
            # 记录当前使用的工具（用于延续对话）
            _last_tool_view = forced_tool

            # 直接执行工具
            if forced_tool == "image_generate":
                # 文生图特殊处理
                result = _call_siliconflow_txt2img_full(message)
                reply = result.get("reply", "")
                _record_history(forced_tool, message, reply)
                return jsonify(result)
            else:
                # 其他工具通过 _execute_tool 执行
                reply = _execute_tool(forced_tool, message)
                return jsonify({"reply": reply})

        # 2.2 首页 → 使用 Agent 引擎（智能路由）
        # 但先检查是否是延续性回复（从工具视图返回后继续对话）
        if _last_tool_view and _last_tool_view in VALID_TOOLS:
            if _is_continuation(message):
                logger.info(f"[Continuation] 步骤 {_continuation_step+1}/{MAX_CONTINUATION_STEPS}, 工具: {_last_tool_view}, 输入: {message}")
                _continuation_step += 1
                reply = _execute_tool(_last_tool_view, message)
                # 检查工具回复是否已完成流程，如果完成则自动重置延续状态
                _check_completion(reply)
                return jsonify({"reply": reply})

        # 2.3 首页 → 使用 Agent 引擎（智能路由）
        _ensure_agent_connected()
        try:
            reply = _agent_engine.process_sync(message)

            # 尝试从 Agent 决策中提取工具名并记录历史
            tool_found = False
            try:
                if len(_agent_engine.conversation_history) >= 2:
                    for entry in reversed(_agent_engine.conversation_history[-5:]):
                        if entry["role"] == "assistant":
                            try:
                                decision = _json.loads(entry["content"])
                                if isinstance(decision, dict) and decision.get("tools"):
                                    for tool in decision["tools"]:
                                        tn = tool.get("name", "")
                                        if tn in VALID_TOOLS:
                                            _record_history(tn, message, reply[:200])
                                            # 记录最后使用的工具
                                            _last_tool_view = tn
                                            tool_found = True
                                            break
                            except:
                                pass
                            break
            except Exception as e:
                logger.debug(f"历史记录失败: {e}")

            # 如果没有找到工具，尝试从回复内容中提取工具名
            if not tool_found:
                for tool_name, label in VALID_TOOLS.items():
                    if label in reply or tool_name.replace("_query", "") in reply.lower():
                        _record_history(tool_name, message, reply[:200])
                        _last_tool_view = tool_name
                        break

            # 清除 Agent 内部对话历史，避免 LLM 被上轮工具带偏
            _agent_engine.conversation_history.clear()

            return jsonify({"reply": reply})

        except Exception as e:
            logger.error(f"[Agent] 处理失败: {e}")
            # 降级到关键词路由
            tool_name = _keyword_decide_tool(message)
            if tool_name != "UNKNOWN":
                reply = _execute_tool(tool_name, message)
                _last_tool_view = tool_name
                return jsonify({"reply": reply})
            return jsonify({
                "reply": "抱歉，我无法理解您的需求。我可以帮您：记账、日程管理、文生图、基金查询、招股书查询。"
            })

    except Exception as e:
        logger.exception(f"[Chat] 处理异常: {e}")
        return jsonify({"reply": f"处理失败: {str(e)[:100]}"}), 500


def _handle_image_upload(image_b64: str, message: str, view: str):
    """
    处理图片上传请求
    """
    global _last_tool_view
    _last_tool_view = "image_generate"

    img_bytes = _b64.b64decode(image_b64)

    # ===== 1. 人脸旋转/图片编辑请求（使用豆包 API）=====
    rot_keywords = ["旋转", "视角", "角度", "侧", "正面", "右转", "左转", "face", "rotate"]
    if any(kw in message.lower() for kw in rot_keywords):
        try:
            logger.info("[Image] 调用豆包进行图片编辑")
            img_b64_str = _b64.b64encode(img_bytes).decode()
            headers = {"Authorization": f"Bearer {DOUBAO_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": DOUBAO_ENDPOINT,
                "prompt": message,
                "image": f"data:image/png;base64,{img_b64_str}",
                "size": DOUBAO_IMAGE_SIZE,
                "n": 1,
            }
            response = http_requests.post(
                f"{DOUBAO_BASE_URL}/images/generations",
                headers=headers, json=payload, timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                img_url = data.get("data", [{}])[0].get("url", "")
                if img_url:
                    img_resp = http_requests.get(img_url, timeout=30)
                    if img_resp.status_code == 200 and len(img_resp.content) > 1000:
                        result_b64 = _b64.b64encode(img_resp.content).decode()
                        try:
                            img_dir = os.path.join(project_dir, "static", "generated")
                            os.makedirs(img_dir, exist_ok=True)
                            filename = f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            with open(os.path.join(img_dir, filename), "wb") as f:
                                f.write(img_resp.content)
                        except Exception as e:
                            logger.warning(f"[Image] 保存失败: {e}")
                        _record_history("image_generate", message, "图片编辑完成")
                        return jsonify({"reply": "图片编辑完成", "images": [result_b64]})
            logger.warning(f"[Image] 豆包编辑返回: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"[Image] 豆包编辑失败: {e}")

        # 后端失败时降级到 PIL 基础旋转
        try:
            from PIL import Image as PILImage
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            angle = 30
            for kw in ["30", "45", "60", "90", "15", "180", "270"]:
                if kw in message:
                    angle = int(kw)
                    break
            if "右转" in message or "right" in message.lower():
                angle = -angle
            rotated = img.rotate(angle, expand=True, fillcolor=(127, 127, 127))
            buf = io.BytesIO()
            rotated.save(buf, format="PNG")
            img_b64 = _b64.b64encode(buf.getvalue()).decode()
            try:
                img_dir = os.path.join(project_dir, "static", "generated")
                os.makedirs(img_dir, exist_ok=True)
                filename = f"rotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                rotated.save(os.path.join(img_dir, filename))
                logger.info(f"[Image] 旋转图片已保存: {filename}")
            except Exception as e:
                logger.warning(f"[Image] 保存失败: {e}")
            _record_history("image_generate", message, f"图片已旋转 {angle} 度")
            return jsonify({
                "reply": f"图片已向左旋转 {abs(angle)} 度",
                "images": [img_b64]
            })
        except Exception as e:
            logger.warning(f"[Image] PIL旋转失败: {e}")
            _record_history("image_generate", message, "旋转失败")
            return jsonify({"reply": f"图片旋转失败: {str(e)[:100]}"})

    # ===== 2. 图片编辑请求（使用 Qwen-Image-Edit） =====
    if message:
        api_key = LLM_CONFIG.get("api_key", "")
        base_url = LLM_CONFIG.get("base_url", "https://api.siliconflow.cn/v1")

        if api_key and not (api_key.startswith("sk-") and len(api_key) < 20):
            try:
                from PIL import Image as PILImage

                # 压缩图片
                pil_img = PILImage.open(io.BytesIO(img_bytes))
                if max(pil_img.size) > 1536:
                    pil_img.thumbnail((1536, 1536))

                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                img_b64 = _b64.b64encode(buf.getvalue()).decode()

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "Qwen/Qwen-Image-Edit-2509",
                    "prompt": message,
                    "image": f"data:image/png;base64,{img_b64}",
                    "image_size": "1024x1024",
                }

                response = http_requests.post(
                    f"{base_url}/image/generations",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()

                images = data.get("images", [])
                if images:
                    img_url = images[0].get("url", "")
                    img_response = http_requests.get(img_url, timeout=30)
                    result_b64 = _b64.b64encode(img_response.content).decode()
                    _record_history("image_generate", message, "图片编辑完成")
                    return jsonify({
                        "reply": f"已处理: {message}",
                        "image": result_b64
                    })
            except Exception as e:
                logger.error(f"[Image] 编辑失败: {e}")
                return jsonify({"reply": f"图片编辑失败: {str(e)[:80]}"})
        else:
            return jsonify({"reply": "请先在 config.yaml 中配置 API Key"})

    # ===== 3. 仅上传图片，无文本 =====
    return jsonify({"reply": "已收到图片，请输入文字描述要求"})


# ============================================================
# 9. 异步图片生成（资源密集型任务）
# ============================================================

_async_tasks: Dict[str, dict] = {}
_async_tasks_lock = threading.Lock()


@app.route("/generate_image_async", methods=["POST"])
def generate_image_async():
    """
    异步图片生成：立即返回 task_id，后台线程处理
    """
    data = request.get_json(silent=True) or {}
    prompt = (data.get("message") or "").strip()

    if not prompt:
        return jsonify({"error": "请输入图片描述"}), 400

    task_id = str(uuid.uuid4())[:8]

    with _async_tasks_lock:
        _async_tasks[task_id] = {"status": "pending", "result": None, "created_at": datetime.now()}

    def _bg_generate(tid: str, msg: str):
        try:
            result = _call_siliconflow_txt2img_full(msg)
            with _async_tasks_lock:
                _async_tasks[tid]["status"] = "done"
                _async_tasks[tid]["result"] = result
        except Exception as e:
            logger.error(f"[Async] 任务 {tid} 失败: {e}")
            with _async_tasks_lock:
                _async_tasks[tid]["status"] = "error"
                _async_tasks[tid]["result"] = {"reply": f"生成失败: {str(e)[:100]}"}

    thread = threading.Thread(target=_bg_generate, args=(task_id, prompt), daemon=True)
    thread.start()

    return jsonify({
        "task_id": task_id,
        "status": "pending",
        "message": "图片正在生成中..."
    })


@app.route("/image_status/<task_id>", methods=["GET"])
def image_status(task_id):
    """查询异步图片生成状态"""
    with _async_tasks_lock:
        task = _async_tasks.get(task_id)

    if not task:
        return jsonify({"error": "任务不存在"}), 404

    response = {"status": task["status"]}
    if task["result"]:
        if isinstance(task["result"], dict):
            response.update(task["result"])
        else:
            response["reply"] = str(task["result"])

    return jsonify(response)


@app.route("/image_async_demo")
def image_async_demo():
    """返回异步图片生成的演示页"""
    return render_template_string("""
    <html><body style="background:#1a1a2e;color:white;font-family:sans-serif;padding:30px">
    <h2>🎨 异步图片生成（资源密集型任务）</h2>
    <form onsubmit="gen(event)">
        <input id="p" style="width:400px;padding:8px;border-radius:6px" placeholder="描述图片内容...">
        <button type="submit" style="padding:8px 20px;background:#e8a838;border:none;border-radius:6px;cursor:pointer">生成</button>
    </form>
    <div id="status" style="margin-top:12px;color:#aaa"></div>
    <div id="img" style="margin-top:12px"></div>
    <script>
    async function gen(e){
        e.preventDefault();const p=document.getElementById('p').value;const st=document.getElementById('status');
        st.textContent='⏳ 提交中...';
        const r=await fetch('/generate_image_async',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:p})});
        const d=await r.json();
        st.textContent='⏳ 生成中 (task: '+d.task_id+')...';
        const poll=setInterval(async()=>{
            const r2=await fetch('/image_status/'+d.task_id);
            const d2=await r2.json();
            if(d2.status==='done'){
                clearInterval(poll);
                st.textContent='✅ 生成完成';
                if(d2.image) document.getElementById('img').innerHTML='<img src="data:image/png;base64,'+d2.image+'" style="max-width:400px;border-radius:8px">';
                else st.textContent=d2.reply||'完成';
            }else if(d2.status==='error'){
                clearInterval(poll);
                st.textContent='❌ '+(d2.reply||'失败');
            }
        },2000);
    }
    </script>
    </body></html>
    """)


@app.route("/cleanup_tasks", methods=["POST"])
def cleanup_tasks():
    """
    清理过期任务（防止内存泄漏）
    删除超过 1 小时的任务
    """
    with _async_tasks_lock:
        now = datetime.now()
        expired = []
        for tid, task in _async_tasks.items():
            if "created_at" in task:
                age = (now - task["created_at"]).total_seconds()
                if age > 3600:  # 1 小时
                    expired.append(tid)

        for tid in expired:
            del _async_tasks[tid]

    return jsonify({"cleaned": len(expired), "remaining": len(_async_tasks)})


@app.route("/favicon.ico")
def favicon():
    """返回空 favicon"""
    return "", 204


# ============================================================
# 10. 启动入口
# ============================================================

def main():
    port = 6001
    logger.info(f"启动 SmartAgent Web UI: http://127.0.0.1:{port}")
    logger.info(f"MCP 状态: {'已连接' if mcp_connected else '待机'}")
    logger.info(f"Agent 状态: {'就绪' if _agent_ready else '初始化中'}")

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )


if __name__ == "__main__":
    main()