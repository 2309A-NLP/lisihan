"""
智能体 Agent 核心引擎 — MCP Client 版本
工单编号：人工智能NLP-Agent数字人项目-智能体任务

通过 Prompt 驱动 LLM 进行意图识别，然后通过 MCP 协议
动态发现工具并调用，取代硬编码的工具适配器。
"""

import asyncio
import json
import logging
import os
import sys
import textwrap
from typing import Dict, List, Optional

# MCP 客户端
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

# 鲁棒 JSON 解析
from core.json_utils import extract_decision

logger = logging.getLogger("Agent")


class Agent:
    """
    基于 MCP 协议 + LLM Prompt 驱动的智能体引擎

    工作流程：
    1. 接收用户输入
    2. 用 LLM 进行意图识别（是否需要工具、哪些工具、参数）
    3. 调用 MCP 工具（支持并行）
    4. 用 LLM 整合结果（多工具）或直接返回（单工具）
    """

    # ========== Prompt 模板 ==========
    PROMPT_TEMPLATE = textwrap.dedent("""\
    你是一个严格的意图分类器+工具调度器。你的核心任务：

    **第一步：判断意图分类（严格遵循以下规则）**
    - **文生图 (image_generate)**: 用户要求"画"、"画一幅"、"生成图片"、"帮我画"、"绘制"等视觉创作请求。**注意：只要用户说"画"某物，就是文生图，绝不要询问日期。**
    - **记账本 (ledger_query)**: 用户涉及"记账"、"花了多少钱"、"收入"、"支出"、"流水"、"账单"等财务记录。**注意："记录"、"确认"是动作词，不是记账事由。**
    - **日程提醒 (schedule_query)**: 用户涉及"提醒"、"会议"、"安排"、"日程"、"日历"等时间管理。
    - **基金 (fund_query)**: 用户查询"基金"、"净值"、"收益率"、"股票"、"涨幅"等投资信息。
    - **招股书 (prospectus_query)**: 用户查询"招股书"、"公司的营收"等招股信息。
    - **其他**: 问候、闲聊、无明确工具需求时，设置 need_tools=false。

    **第二步：实体抽取规则（关键！）**
    - 动作词"记录"、"确认"、"删除"、"修改"、"查"是操作指令，**不是**记账事由（reason）。
    - 记账事由应该是"午餐"、"交通"、"买菜"等具体消费项目。
    - 日程事由应该是"开会"、"牙医"、"聚餐"等具体事件。
    - 文生图描述不要抽取时间、日期等无关信息。

    **第三步：输出决策**
    - **最多选择一个工具，不要同时选择多个。** 如果感觉多个工具都可能用到，选最贴切的那个。
    - 如果用户明确要数据查询，但**没有要求**生成图表，**不要自动调用** image_generate。
      你可以在回答的最后询问"需要我生成一张示意图吗？"来把决定权交给用户。
    - 如果有明确的工具意图，必须设置 need_tools=true，direct_reply=""
    - 只有问候、闲聊、不需要工具时才设置 need_tools=false
    - 不要解释"可以通过什么工具查询"——直接调用工具即可
    - 直接回复必须用中文

    ## 可用工具
    {tool_descriptions}

    ## 对话历史
    {conversation_history}

    ## 用户输入
    {user_input}

    ## 输出格式（纯JSON，不要输出其他内容）
    {{"need_tools":true/false,"tools":[{{"name":"工具名","args":{{"param":"值"}}}}],"direct_reply":"直接回复"}}
    """)

    RESULT_INTEGRATION_PROMPT = textwrap.dedent("""\
    将工具返回结果整理成自然语言回复。

    问题: {user_input}

    结果:
    {tool_results}

    ## 输出（纯JSON）
    {{"reply":"你的回答"}}
    """)

    def __init__(self, mcp_server_script: str, llm_config: dict = None):
        """
        初始化 Agent

        Args:
            mcp_server_script: MCP 服务器脚本路径
            llm_config: LLM 配置（api_key, base_url, model, temperature等）
        """
        self.mcp_server_script = os.path.abspath(mcp_server_script)
        self.llm_config = llm_config or {}
        self.session: Optional[ClientSession] = None          # MCP 客户端会话
        self._tools_cache: List[dict] = []                   # 工具列表缓存
        self.conversation_history: List[Dict] = []           # 对话历史
        self._last_user_input: str = ""                      # 最近一次用户输入
        self._loop = None                                    # 持久化事件循环

        # 配置超时参数
        self.timeouts = {
            "intent_recognition": 8,      # 意图识别超时（秒）
            "tool_call": 120,             # 单个工具调用超时（RAG慢查询需要2分钟）
            "integration": 10,            # 结果整合超时
            "image_generate": 60,         # 图片生成超时
        }

    # ========== MCP 连接管理 ==========

    async def connect(self):
        """
        启动 MCP 服务器子进程并建立会话

        如果 MCP 连接失败，降级到 HTTP 直连模式（无需子进程）
        """
        try:
            # 配置 MCP 服务器启动参数
            server_params = StdioServerParameters(
                command=sys.executable,          # Python 解释器
                args=[self.mcp_server_script],   # 服务器脚本
            )

            # 建立 stdio 管道
            read, write = await stdio_client(server_params).__aenter__()

            # 创建客户端会话并初始化
            self.session = await ClientSession(read, write).__aenter__()
            await self.session.initialize()

            # 获取工具列表并缓存
            tools_result = await self.session.list_tools()
            self._tools_cache = []
            for t in tools_result.tools:
                self._tools_cache.append({
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, 'inputSchema') else {},
                })
            logger.info(f"已发现 {len(self._tools_cache)} 个 MCP 工具")

        except Exception as e:
            logger.warning(f"MCP连接失败，使用HTTP直连: {e}")
            self.session = None

    async def disconnect(self):
        """断开 MCP 连接，释放资源"""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass
            self.session = None

    # ========== 工具描述生成 ==========

    def _describe_tools(self) -> str:
        """
        生成工具描述文本，用于 Prompt

        Returns:
            格式化的工具描述字符串
        """
        if self._tools_cache:
            lines = []
            for t in self._tools_cache:
                schema = t["inputSchema"]
                props = schema.get("properties", {}) if schema else {}
                params = []
                for pname, pinfo in props.items():
                    required = pname in schema.get("required", [])
                    mark = " (必填)" if required else ""
                    params.append(f"    - {pname}{mark}: {pinfo.get('description', '')}")
                param_str = "\n".join(params) if params else "    无参数"
                lines.append(f"- **{t['name']}**: {t['description']}\n  参数:\n{param_str}")
            return "\n\n".join(lines)

        # MCP未连接时使用默认描述（降级方案）
        return textwrap.dedent("""\
        - **ledger_query**: 记账本，查询收支记录、消费流水
        - **schedule_query**: 日程提醒，查询日程安排、提醒设置
        - **image_generate**: 文生图，根据文本生成图片
        - **fund_query**: 基金数据问答，查询基金净值、收益率
        - **prospectus_query**: 招股说明书问答，查询招股书中的财务数据、公司信息
        """)

    def _format_history(self) -> str:
        """格式化对话历史（最近6轮）用于 Prompt"""
        if not self.conversation_history:
            return "暂无历史对话"
        lines = []
        for msg in self.conversation_history[-6:]:  # 只取最近6条
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(lines)

    def _fallback_tools(self, user_input: str) -> list:
        """
        关键词回退：从匹配的工具中选出得分最高的一个

        如果有多个工具匹配，选得分最高的（最高=匹配关键词最多的）。
        如果得分相同，按优先级：fund_query > ledger_query > schedule_query
        > prospectus_query > image_generate。

        Args:
            user_input: 用户输入文本

        Returns:
            最多一个工具（或空列表）
        """
        text = user_input.lower()
        keyword_map = {
            "ledger_query": ["记账", "记录", "账本", "收支", "花费", "支出", "收入", "花了", "流水"],
            "schedule_query": ["日程", "提醒", "会议", "安排", "日历", "明天", "后天"],
            "image_generate": ["生成图片", "生成一张", "画一幅", "画一张", "帮我画", "绘制", "画图", "作图", "文生图"],
            "fund_query": ["基金", "净值", "收益率", "fund", "股票", "涨幅", "涨跌", "代码"],
            "prospectus_query": ["招股", "说明书", "prospectus", "公司", "营收", "财务数据", "募投", "配售", "ipo", "首发"],
        }
        # 优先级排序（同分时使用）
        priority_order = ["fund_query", "ledger_query", "schedule_query", "prospectus_query", "image_generate"]

        scores = {}
        for tool, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[tool] = score

        if not scores:
            return []

        # 找最高分
        max_score = max(scores.values())
        best = [t for t, s in scores.items() if s == max_score]

        # 同分时按优先级选
        for tool in priority_order:
            if tool in best:
                return [{"name": tool, "args": {}}]

        return [{"name": best[0], "args": {}}]

    # ========== LLM 调用 ==========

    def _call_llm(self, prompt: str, timeout_sec: int = 10) -> str:
        """
        调用 LLM（OpenAI 兼容接口），带超时控制

        Args:
            prompt: 提示词
            timeout_sec: 超时时间（秒）

        Returns:
            LLM 返回的文本

        Raises:
            ValueError: API Key 未配置
            requests.RequestException: API 调用失败
        """
        import requests

        # 获取配置
        api_key = self.llm_config.get("api_key", "")
        if not api_key or (api_key.startswith("sk-") and len(api_key) < 20):
            # 检查是否为示例 key
            raise ValueError("LLM API Key 未配置或为示例值，请在 config/config.yaml 中设置")

        base_url = self.llm_config.get("base_url", "https://api.siliconflow.cn/v1")
        model = self.llm_config.get("model", "Qwen/Qwen2.5-7B-Instruct")
        temperature = self.llm_config.get("temperature", 0.1)
        max_tokens = min(self.llm_config.get("max_tokens", 1024), 2048)

        # 构建请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 发起请求
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_sec,
        )
        resp.raise_for_status()

        # 提取返回内容
        return resp.json()["choices"][0]["message"]["content"]

    # ========== 工具调用 ==========

    # 豆包（火山引擎）文生图配置
    DOUBAO_API_KEY = "ark-7419c003-7697-4d54-bbec-f08ad08094e5-25ec6"
    DOUBAO_ENDPOINT = "ep-20260625141102-6t8th"
    DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    # 工具后端API直接调用配置（不依赖 MCP 子进程）
    TOOL_API_ENDPOINTS = {
        "ledger_query": {"url": "http://127.0.0.1:8081/api/chat", "param": "message"},
        "schedule_query": {"url": "http://127.0.0.1:5000/chat", "param": "message"},
        "fund_query": {"url": "http://127.0.0.1:5002/ask", "param": "question"},
        "prospectus_query": {"url": "http://127.0.0.1:5003/ask", "param": "question"},
    }

    async def _call_tool_direct(self, name: str, args: dict) -> str:
        """
        通过 HTTP API 直接调用工具（无需 MCP 子进程）

        这是降级方案，当 MCP 不可用时使用
        """
        import requests

        # ===== 文生图特殊处理 =====
        if name == "image_generate":
            param = args.get("prompt") or args.get("param") or args.get("query") or self._last_user_input
            if not param:
                return "未提供图片生成描述"

            # 优先使用豆包（火山引擎 Doubao-Seedream-4.5）
            try:
                logger.info(f"[Agent] 调用豆包文生图: {param[:30]}...")
                import requests as _req
                h = {"Authorization": f"Bearer {self.DOUBAO_API_KEY}", "Content-Type": "application/json"}
                p = {"model": self.DOUBAO_ENDPOINT, "prompt": param, "size": "1920x1920", "n": 1}
                r = _req.post(f"{self.DOUBAO_BASE_URL}/images/generations", headers=h, json=p, timeout=60)
                if r.status_code == 200:
                    img_url = r.json().get("data", [{}])[0].get("url", "")
                    if img_url:
                        return f"图片已生成: {img_url}"
                logger.warning(f"[Agent] 豆包返回: HTTP {r.status_code}")
            except Exception as e:
                logger.warning(f"[Agent] 豆包失败: {e}")

            # 降级到 SiliconFlow
            try:
                api_key = self.llm_config.get("api_key", "")
                base_url = self.llm_config.get("base_url", "https://api.siliconflow.cn/v1")
                h2 = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                p2 = {"model": "black-forest-labs/FLUX.1-dev", "prompt": param, "image_size": "1024x1024", "num_inference_steps": 4}
                r2 = _req.post(f"{base_url}/image/generations", headers=h2, json=p2, timeout=60)
                r2.raise_for_status()
                imgs = r2.json().get("images", [])
                if imgs:
                    return f"图片已生成: {imgs[0].get('url', '')}"
                return "图片生成完成"
            except Exception as e:
                return f"图片生成失败: {e}"

        # ===== 普通工具调用 =====
        endpoint = self.TOOL_API_ENDPOINTS.get(name)
        if not endpoint:
            return f"未知工具: {name}"

        # 构建参数：优先使用 LLM 提取的参数，否则使用用户原始输入
        param_value = args.get(endpoint["param"]) or args.get("query") or args.get("param") or self._last_user_input or ""
        payload = {endpoint["param"]: param_value}

        try:
            resp = requests.post(
                endpoint["url"],
                json=payload,
                timeout=self.timeouts["tool_call"]
            )
            resp.raise_for_status()
            result = resp.json()
            # 提取回复字段（不同工具返回格式不同）
            return result.get("reply") or result.get("answer") or result.get("text") or str(result)
        except requests.ConnectionError:
            return f"【{name}】后端服务未启动，请先启动 {endpoint['url']}"
        except requests.Timeout:
            return f"【{name}】后端服务响应超时（{self.timeouts['tool_call']}s），请检查服务状态"
        except requests.RequestException as e:
            return f"【{name}】调用失败: {e}"

    async def _call_mcp_tool(self, name: str, args: dict) -> str:
        """
        通过 MCP 协议调用工具，失败时自动降级到 HTTP 直连

        Args:
            name: 工具名
            args: 工具参数

        Returns:
            工具执行结果（字符串）
        """
        if self.session:
            try:
                # 构建参数：合并 LLM 提取的参数和用户原始输入
                mcp_args = args.copy() if args else {}

                # 如果没有指定参数，使用用户原始输入
                if not mcp_args:
                    mcp_args["query"] = self._last_user_input or ""

                # 自动适配参数名：不同工具使用不同的参数名
                # fund_query 和 prospectus_query 使用 "question"
                if name in ["fund_query", "prospectus_query"]:
                    if "query" in mcp_args and "question" not in mcp_args:
                        mcp_args["question"] = mcp_args.pop("query")
                    elif "question" not in mcp_args:
                        mcp_args["question"] = self._last_user_input or ""

                # ledger_query 和 schedule_query 使用 "query"
                if name in ["ledger_query", "schedule_query"]:
                    if "query" not in mcp_args:
                        mcp_args["query"] = self._last_user_input or ""

                # image_generate 使用 "prompt"（MCP 工具定义中参数名为 prompt）
                if name == "image_generate":
                    if "query" in mcp_args and "prompt" not in mcp_args:
                        mcp_args["prompt"] = mcp_args.pop("query")
                    elif "prompt" not in mcp_args:
                        mcp_args["prompt"] = self._last_user_input or ""

                result: CallToolResult = await self.session.call_tool(name, mcp_args)

                if result.isError:
                    error_msg = result.content[0].text if result.content else '未知错误'
                    return f"工具执行出错: {error_msg}"

                # 提取返回内容
                parts = []
                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        parts.append(item.text)
                    elif hasattr(item, "data") and item.data:
                        parts.append(f"[二进制数据: {len(item.data)} bytes]")
                return "\n".join(parts) if parts else "工具执行完成（无返回内容）"

            except Exception as e:
                logger.warning(f"MCP工具 {name} 调用失败，降级到 HTTP 直连: {e}")

        # 降级到 HTTP 直连
        return await self._call_tool_direct(name, args)

    async def _call_mcp_tools_parallel(self, tool_calls: list) -> list:
        """
        并行调用多个 MCP 工具

        Args:
            tool_calls: 工具调用列表 [{"name": "xxx", "args": {...}}, ...]

        Returns:
            结果列表（字符串）
        """
        tasks = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tasks.append(self._call_mcp_tool(name, args))

        # 并行执行，捕获异常
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 格式化输出
        output = []
        for tc, result in zip(tool_calls, results):
            name = tc["name"]
            if isinstance(result, Exception):
                output.append(f"【{name}】调用失败: {result}")
            else:
                output.append(f"【{name}】\n{result}")
        return output

    # ========== 主处理流程 ==========

    async def process(self, user_input: str) -> str:
        """
        处理用户输入 — 优化版

        流程：
        1. 意图识别（LLM / 关键词回退）
        2. 工具调用（并行/串行）
        3. 结果整合（单工具跳过LLM，多工具用LLM整合）

        Args:
            user_input: 用户输入文本

        Returns:
            回复文本
        """
        import time as _time
        t_start = _time.time()

        # 保存用户输入，供工具调用使用
        self._last_user_input = user_input
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # ===== 先检查问候语/闲聊（不调用 LLM，节省成本并提高响应速度） =====
            greetings = [
                "你好", "您好", "hi", "hello", "hey",
                "在吗", "在不在", "在吗？",
                "help", "帮助", "帮忙",
                "你是谁", "你是什么", "你能做什么", "有什么功能",
                "功能", "菜单", "指令",
                "谢谢", "感谢", "再见", "拜拜"
            ]
            if any(g in user_input.lower()[:50] for g in greetings):
                reply = "你好！请问有什么需要帮忙的？我可以帮你记账、管理日程、生成图片、查询基金和招股书。"
                self.conversation_history.append({"role": "assistant", "content": reply})
                elapsed = _time.time() - t_start
                logger.info(f"[PERF] 总耗时: {elapsed:.2f}s (问候语)")
                return reply

            # ========== 阶段1: 意图识别 ==========
            prompt = self.PROMPT_TEMPLATE.format(
                tool_descriptions=self._describe_tools(),
                conversation_history=self._format_history(),
                user_input=user_input,
            )

            t1 = _time.time()
            try:
                llm_raw = self._call_llm(prompt, timeout_sec=self.timeouts["intent_recognition"])
                logger.debug(f"[DEBUG] LLM 原始返回: {llm_raw[:500]}")
            except Exception as e:
                logger.warning(f"LLM意图识别超时/失败: {e}，使用关键词回退")
                tools = self._fallback_tools(user_input)
                if tools:
                    llm_raw = f'{{"need_tools":true,"tools":{json.dumps(tools)},"direct_reply":""}}'
                else:
                    llm_raw = '{"need_tools":false,"direct_reply":"你好！请问有什么需要帮忙的？"}'

            t_intent = _time.time() - t1
            logger.info(f"[PERF] 意图识别: {t_intent:.2f}s | RAW: {llm_raw[:200]}")

            # 解析 LLM 返回
            decision = extract_decision(llm_raw)

            # ===== 安全兜底：LLM 有时会错误地把工具调用写成 direct_reply =====
            # 例如：{"need_tools":false,"direct_reply":"可以通过【prospectus_query】工具查询"}
            # 此时 direct_reply 中包含工具名，说明 LLM 本意是要调工具
            reply_text = decision.get("direct_reply", "")
            if not decision.get("need_tools", False) and reply_text:
                for tool_key in ["ledger_query", "schedule_query", "image_generate",
                                  "fund_query", "prospectus_query"]:
                    if tool_key in reply_text:
                        logger.info(f"[SafeGuard] LLM 把工具调用写成了回复，强制修正: {tool_key}")
                        decision["need_tools"] = True
                        decision["tools"] = [{"name": tool_key, "args": {}}]
                        decision["direct_reply"] = ""
                        break

            # 不需要工具：直接回复
            if not decision.get("need_tools", False):
                reply = decision.get("direct_reply", "你好！请问有什么需要帮忙的？")
                elapsed = _time.time() - t_start
                self.conversation_history.append({"role": "assistant", "content": reply})
                logger.info(f"[PERF] 总耗时: {elapsed:.2f}s (无需工具)")
                return reply

            # ========== 阶段2: 工具调用 ==========
            tool_calls = decision.get("tools", [])
            if not tool_calls:
                tool_calls = self._fallback_tools(user_input)

            if not tool_calls:
                reply = "抱歉，我不太理解您的需求，请换个方式描述。"
                elapsed = _time.time() - t_start
                self.conversation_history.append({"role": "assistant", "content": reply})
                logger.info(f"[PERF] 总耗时: {elapsed:.2f}s (未识别到工具)")
                return reply

            t2 = _time.time()

            # ── 单工具（主流路径）：干净调用，无前缀包装 ──
            if len(tool_calls) == 1:
                tc = tool_calls[0]
                result = await self._call_mcp_tool(tc["name"], tc.get("args", {}))
                raw_reply = result.strip()

                # 文生图：直接返回结果
                if tc["name"] == "image_generate":
                    reply = raw_reply
                else:
                    # 数据查询类：干净返回，末尾询问是否需要生成示意图
                    reply = raw_reply
                    data_tools = {"fund_query", "ledger_query", "schedule_query", "prospectus_query"}
                    # 仅在用户没有主动要求生成图片时询问
                    if tc["name"] in data_tools and not any(kw in user_input for kw in ["图", "图片", "画", "截图"]):
                        reply += "\n\n如果需要，我可以为您生成一张示意图。"

            # ── 多工具（极少发生）：并行调用 + LLM整合 ──
            else:
                raw_results = await self._call_mcp_tools_parallel(tool_calls)
                tool_text = "\n\n".join(raw_results)
                integration_prompt = self.RESULT_INTEGRATION_PROMPT.format(
                    user_input=user_input,
                    tool_results=tool_text,
                )
                t3 = _time.time()
                try:
                    integrate_raw = self._call_llm(
                        integration_prompt,
                        timeout_sec=self.timeouts["integration"]
                    )
                    integrate_result = extract_decision(integrate_raw)
                    reply = integrate_result.get("reply", tool_text)
                except Exception as e:
                    logger.warning(f"结果整合失败，使用原始结果: {e}")
                    reply = tool_text
                t_integrate = _time.time() - t3
                logger.info(f"[PERF] 结果整合: {t_integrate:.2f}s")

            # 保存到历史
            self.conversation_history.append({"role": "assistant", "content": reply})
            elapsed = _time.time() - t_start
            logger.info(f"[PERF] 总耗时: {elapsed:.2f}s (tools={len(tool_calls)})")
            return reply

        except Exception as e:
            elapsed = _time.time() - t_start
            # 确保异常信息完整
            err_msg = str(e) or repr(e) or type(e).__name__
            logger.exception(f"处理失败 ({elapsed:.2f}s): {err_msg}")
            return f"处理失败：{err_msg[:200]}"

    def process_sync(self, user_input: str) -> str:
        """
        同步包装器 — 使用持久化的事件循环

        解决跨循环 session 冲突问题

        Args:
            user_input: 用户输入

        Returns:
            回复文本
        """
        if not hasattr(self, '_loop') or self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        try:
            return self._loop.run_until_complete(self.process(user_input))
        except Exception as e:
            import traceback
            err_msg = str(e) or repr(e) or type(e).__name__
            err_detail = f"{type(e).__name__}: {err_msg}"
            logger.error(f"process_sync 失败: {err_detail}\n{traceback.format_exc()}")
            return f"处理失败: {err_msg[:200]}"

    def sync_connect(self):
        """
        同步方式建立 MCP 连接（使用持久化 event loop）
        """
        if not hasattr(self, '_loop') or self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        if not self.session:
            self._loop.run_until_complete(self.connect())


# ===================== 命令行入口 =====================

def main():
    """命令行交互入口"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 确定文件路径
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mcp_script = os.path.join(project_dir, "mcp_server.py")
    config_path = os.path.join(project_dir, "config", "config.yaml")

    # 加载配置
    llm_config = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
            llm_config = full.get("llm", {})
    else:
        print(f"[!] 配置文件不存在: {config_path}")

    # 检查 API Key
    api_key = llm_config.get("api_key", "")
    if not api_key or (api_key.startswith("sk-") and len(api_key) < 20):
        print("[!] Please set your LLM API Key in config/config.yaml first")
        print("    File: C:\\Users\\freedom\\Desktop\\06-Agent智能体项目\\config\\config.yaml")
        print("    Edit: llm.api_key = your SiliconFlow / OpenAI Key\n")

    # 创建 Agent
    agent = Agent(mcp_script, llm_config)

    # 交互界面
    print("=" * 50)
    print("  智能管家 Agent v2.0 (MCP)")
    print("  工单编号：人工智能NLP-Agent数字人项目-智能体任务")
    print("=" * 50)
    print()

    async def cli_loop():
        """异步命令行循环"""
        await agent.connect()
        print(f"[MCP] Ready, {len(agent._tools_cache)} tools found\n")
        for t in agent._tools_cache:
            print(f"  [tool] {t['name']}: {t['description'][:60]}")
        print()

        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() in ("exit", "quit"):
                    print("Bye!")
                    break
                if not user_input.strip():
                    continue

                reply = await agent.process(user_input)
                print(f"\n>> {reply}\n")

            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")

        await agent.disconnect()

    asyncio.run(cli_loop())


if __name__ == "__main__":
    main()