"""
鲁棒 JSON 提取工具 — 对 LLM 不规范的 JSON 输出进行修复和提取

工单编号：人工智能NLP-Agent数字人项目-智能体任务

设计思路：
1. 先尝试括号平衡提取
2. 多策略修复常见 JSON 错误
3. 手动解析（正则兜底）
4. 关键词回退（最终保底）

确保即使 LLM 输出格式不标准，也能提取出可用信息
"""

import json
import re
import logging
from typing import Dict, List, Optional, Union, Set

logger = logging.getLogger("Agent.JSON")

# 已知工具名集合（使用列表保证顺序）
TOOL_NAMES = [
    "ledger_query",
    "schedule_query",
    "image_generate",
    "fund_query",
    "prospectus_query"
]
TOOL_NAMES_SET = set(TOOL_NAMES)  # 用于快速查找

# 将工具名映射到友好标签
TOOL_LABELS = {
    "ledger_query": "记账本",
    "schedule_query": "日程提醒",
    "image_generate": "文生图",
    "fund_query": "基金数据",
    "prospectus_query": "招股说明书",
}


def extract_json_balanced(text: str) -> str:
    """
    使用括号平衡算法提取最外层的合法 JSON 字符串

    策略：逐字符扫描，记录大括号深度，找到匹配的闭合括号

    Args:
        text: 包含 JSON 的原始文本

    Returns:
        提取出的 JSON 字符串，如果失败则返回原文本
    """
    if not text:
        return ""

    text = text.strip()

    # ===== 步骤1: 去除 Markdown 代码块 =====
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        # 去除 "json" 标签
        if text.startswith("json"):
            text = text[4:].strip()

    # ===== 步骤2: 找到第一个 { 并做括号平衡 =====
    start = text.find("{")
    if start < 0:
        return text  # 没有大括号，返回原文本

    depth = 0
    in_string = False
    escape = False
    end = start
    text_len = len(text)

    for i in range(start, text_len):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if depth == 0:
        return text[start:end + 1]
    else:
        # 括号不平衡，返回原文本（后续修复策略会处理）
        return text[start:]


def try_extract_json(text: str) -> dict:
    """
    多策略尝试从 LLM 回复中提取 JSON 对象

    策略链（从严格到宽松）：
    1. 括号平衡 + 直接解析
    2. 常见错误修复
    3. 手动正则提取
    4. 关键词回退

    Args:
        text: LLM 输出的原始文本

    Returns:
        提取的字典，如果全部失败则返回空字典
    """
    if not text:
        return {}

    # ===== 策略1: 直接提取并解析 =====
    balanced = extract_json_balanced(text)
    try:
        return json.loads(balanced)
    except json.JSONDecodeError:
        pass

    # ===== 策略2: 尝试多种修复策略 =====
    for fixed_json in _generate_fixes(balanced):
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            continue

    # ===== 策略3: 手动解析（正则提取） =====
    manual = _manual_parse(text)
    if manual and manual.get("tools"):
        return manual

    # ===== 策略4: 关键词回退（保底） =====
    return _fallback_parse(text)


def _generate_fixes(text: str):
    """
    生成一系列可能的 JSON 修复版本

    Args:
        text: 待修复的 JSON 字符串

    Yields:
        修复后的字符串（迭代器）
    """
    if not text:
        return

    current = text

    # ===== 修复0: 补上缺失的闭合大括号 =====
    if not current.rstrip().endswith("}"):
        current = current.rstrip() + "\n}"
        yield current
    else:
        yield current

    # ===== 修复1: 去除多余逗号（JSON 不允许尾随逗号） =====
    current = re.sub(r',\s*([}\]])', r'\1', current)
    yield current

    # ===== 修复2: 修复重复的布尔值 =====
    # true, true → true
    current = re.sub(r'(true|false)\s*,\s*(true|false)', r'\1', current)
    yield current

    # ===== 修复3: 修复 key 前缺少引号 =====
    # 如 tools": → "tools":
    current = re.sub(r'(?<=[\s,\{])(\w+)"\s*:', r'"\1":', current)
    yield current

    # ===== 修复4: 修复对象 key 缺少引号 =====
    # 如 {tools: [...]} → {"tools": [...]}
    current = re.sub(r'(?<=[,{])\s*(\w+)\s*:', r'"\1":', current)
    yield current

    # ===== 修复5: 修复值后面缺逗号（换行场景） =====
    # 如 true\ntools": → true,\n"tools":
    current = re.sub(r'(true|false|\d+|"[^"]*")\s*\n\s*"', r'\1,\n"', current)
    yield current

    # ===== 修复6: 处理工具名作为外层 key 的异常结构 =====
    # 例如: {"need_tools": true, "fund_query": [...]}
    # 这种结构虽然不合规范，但可以接受
    try:
        d = json.loads(current)
        if isinstance(d, dict):
            # 检查顶层是否有工具名作为 key
            need_tools = d.get("need_tools", True)
            tools = d.get("tools", [])

            # 收集所有工具
            keys_to_move = [key for key in d.keys() if key in TOOL_NAMES_SET]
            for key in keys_to_move:
                val = d.pop(key)
                if isinstance(val, list):
                    tools.extend(val)
                elif isinstance(val, dict):
                    tools.append(val)

            if tools:
                d["tools"] = tools
                d["need_tools"] = bool(need_tools)
                yield json.dumps(d)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass


def _manual_parse(text: str) -> dict:
    """
    手动从文本中提取结构化信息（正则表达式兜底）

    适用于 LLM 输出格式严重错误，但包含关键信息的情况

    Args:
        text: LLM 输出的原始文本

    Returns:
        提取的字典
    """
    result = {"need_tools": True, "tools": [], "direct_reply": ""}

    # ===== 提取 need_tools =====
    need_match = re.search(r'"need_tools"\s*:\s*(true|false)', text, re.IGNORECASE)
    if need_match:
        result["need_tools"] = need_match.group(1).lower() == "true"

    # ===== 提取 direct_reply =====
    reply_match = re.search(r'"direct_reply"\s*:\s*"([^"]+)"', text)
    if reply_match:
        result["direct_reply"] = reply_match.group(1)

    # ===== 提取工具名（多种模式） =====
    seen_names: Set[str] = set()
    found_tools = []

    # 模式1: "name": "xxx", "args": {...}
    pattern1 = r'"name"\s*:\s*"([^"]+)"[^}]*?"args"\s*:\s*\{([^}]*)\}'
    for m in re.finditer(pattern1, text, re.DOTALL):
        name = _normalize_tool_name(m.group(1))
        if name and name not in seen_names:
            seen_names.add(name)
            # 尝试提取参数
            args_str = m.group(2).strip()
            args = {}
            if args_str:
                try:
                    # 尝试解析参数
                    param_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', args_str)
                    for key, val in param_pairs:
                        args[key] = val
                except Exception:
                    pass
            found_tools.append({"name": name, "args": args})

    # 模式2: tools 数组中的 name 字段
    pattern2 = r'name["\s]*:?\s*["\'](ledger_query|schedule_query|image_generate|fund_query|prospectus_query)["\']'
    for m in re.finditer(pattern2, text, re.DOTALL):
        name = m.group(1)
        if name not in seen_names:
            seen_names.add(name)
            found_tools.append({"name": name, "args": {}})

    # 模式3: 工具名作为 key（如 "fund_query": {"name": "fund_query"}）
    pattern3 = r'"([a-z_]+)"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"'
    for m in re.finditer(pattern3, text, re.DOTALL):
        key_name = m.group(1)
        inner_name = m.group(2)
        # 优先使用内部 name
        name = _normalize_tool_name(inner_name) or _normalize_tool_name(key_name)
        if name and name not in seen_names:
            seen_names.add(name)
            found_tools.append({"name": name, "args": {}})

    # 模式4: 直接匹配工具名（作为裸词）
    for tool in TOOL_NAMES:
        if tool in text and tool not in seen_names:
            seen_names.add(tool)
            found_tools.append({"name": tool, "args": {}})

    result["tools"] = found_tools
    return result


def _normalize_tool_name(name: str) -> Optional[str]:
    """
    规范化工具名

    Args:
        name: 原始工具名

    Returns:
        规范后的工具名，如果无法识别则返回 None
    """
    if not name:
        return None

    name = name.strip().lower()

    # 直接匹配
    if name in TOOL_NAMES_SET:
        return name

    # 去掉 _query 后缀后匹配
    if name.endswith("_query"):
        base = name[:-6]  # 去掉 "_query"
        if base + "_query" in TOOL_NAMES_SET:
            return base + "_query"

    # 添加 _query 后缀后匹配
    if name + "_query" in TOOL_NAMES_SET:
        return name + "_query"

    # 特殊处理 image_generate
    if name in ("image", "img", "picture"):
        return "image_generate"

    # 特殊处理 prospectus
    if name in ("prospectus", "招股"):
        return "prospectus_query"

    # 特殊处理 ledger
    if name in ("ledger", "记账"):
        return "ledger_query"

    # 特殊处理 schedule
    if name in ("schedule", "日程"):
        return "schedule_query"

    # 特殊处理 fund
    if name in ("fund", "基金"):
        return "fund_query"

    # 没有匹配
    return None


def _fallback_parse(text: str) -> dict:
    """
    最终回退：基于关键词匹配的工具识别

    这是最后的保底方案，准确率不高但总能返回一个结果

    Args:
        text: 用户输入文本（不是 LLM 输出）

    Returns:
        回退决策字典
    """
    result = {"need_tools": True, "tools": [], "direct_reply": ""}

    text_lower = text.lower() if text else ""

    # ===== 检查是否为问候/闲聊 =====
    greetings = [
        "你好", "您好", "hi", "hello", "hey",
        "在吗", "在不在", "在吗？",
        "help", "帮助", "帮忙",
        "你是谁", "你是什么", "你能做什么", "有什么功能",
        "功能", "菜单", "指令",
        "谢谢", "感谢", "再见", "拜拜"
    ]
    for g in greetings:
        if g in text_lower[:50]:
            result["need_tools"] = False
            result["direct_reply"] = "你好！请问有什么需要帮忙的？"
            return result

    # ===== 关键词映射（扩展版） =====
    keyword_map = {
        # 记账本
        "ledger_query": [
            "记账", "记录", "账本", "收支", "花费", "支出", "收入",
            "花了", "流水", "账单", "消费", "开销",
            "用了多少钱", "花了多少", "余额", "余钱"
        ],
        # 日程提醒
        "schedule_query": [
            "日程", "提醒", "会议", "安排", "日历",
            "明天", "后天", "下周", "本月", "今天",
            "待办", "todo", "计划", "行程"
        ],
        # 文生图 - 注意：不用单字"画""图"，防误匹配
        "image_generate": [
            "生成图片", "生成一张", "画一幅", "画一张",
            "帮我画", "绘制", "画图", "作图",
            "ai画图", "文生图"
        ],
        # 基金数据
        "fund_query": [
            "基金", "净值", "收益率", "fund", "投资",
            "理财", "股票", "涨跌", "收益"
        ],
        # 招股说明书
        "prospectus_query": ["招股", "说明书", "招股说明", "prospectus",
            "公司", "营收", "财务数据", "募投", "上市",
            "财报", "报表", "利润", "收入", "配售", "ipo", "首发"],
    }

    found = set()
    for tool, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            found.add(tool)

    if found:
        result["tools"] = [{"name": t, "args": {}} for t in found]
    elif any(kw in text_lower for kw in ["查", "查询", "问", "找"]):
        # 有查询意图但没识别到具体工具 → 使用泛化工具
        # 尝试通过上下文判断
        if "记账" in text_lower or "账本" in text_lower:
            result["tools"].append({"name": "ledger_query", "args": {}})
        elif "日程" in text_lower or "提醒" in text_lower:
            result["tools"].append({"name": "schedule_query", "args": {}})
        elif "基金" in text_lower or "净值" in text_lower:
            result["tools"].append({"name": "fund_query", "args": {}})
        elif "招股" in text_lower or "说明书" in text_lower:
            result["tools"].append({"name": "prospectus_query", "args": {}})
        elif "画" in text_lower or "生成图片" in text_lower or "图片" in text_lower:
            result["tools"].append({"name": "image_generate", "args": {}})
        else:
            # 默认使用记账本（最常用）
            result["tools"].append({"name": "ledger_query", "args": {}})

    return result


def extract_decision(text: str) -> dict:
    """
    从 LLM 原始输出中提取工具决策（主入口）

    这是整个模块的统一入口，外部调用此函数即可

    Args:
        text: LLM 输出的原始文本（可能包含 JSON 或其他格式）

    Returns:
        标准化的决策字典，包含以下字段：
        - need_tools: bool, 是否需要工具
        - tools: list, 工具调用列表 [{"name": "xxx", "args": {...}}]
        - direct_reply: str, 直接回复（当 need_tools=False 时）
    """
    if not text:
        logger.warning("输入文本为空，返回默认决策")
        return {"need_tools": False, "tools": [], "direct_reply": "请提供有效的输入。"}

    try:
        # 尝试所有策略提取 JSON
        decision = try_extract_json(text)
        if not isinstance(decision, dict):
            logger.warning(f"提取结果不是字典: {type(decision)}，使用回退")
            decision = _fallback_parse(text)
    except Exception as e:
        logger.warning(f"提取失败 ({e})，使用回退")
        decision = _fallback_parse(text)

    # ===== 确保必要字段存在 =====
    # need_tools
    if "need_tools" not in decision:
        decision["need_tools"] = bool(decision.get("tools"))

    # tools
    if "tools" not in decision:
        decision["tools"] = []
    elif not isinstance(decision["tools"], list):
        # 如果 tools 不是列表，尝试转换
        if isinstance(decision["tools"], dict):
            decision["tools"] = [decision["tools"]]
        else:
            decision["tools"] = []

    # 确保每个工具都有 name
    for tool in decision["tools"]:
        if "name" not in tool:
            tool["name"] = "unknown"
        if "args" not in tool:
            tool["args"] = {}

    # direct_reply
    if "direct_reply" not in decision:
        decision["direct_reply"] = ""

    logger.debug(f"决策结果: need_tools={decision['need_tools']}, "
                f"tools={len(decision['tools'])}, "
                f"reply={decision['direct_reply'][:50] if decision['direct_reply'] else '无'}")

    return decision


# ===================== 测试用 =====================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.DEBUG)

    # 测试用例
    test_cases = [
        '{"need_tools": true, "tools": [{"name": "ledger_query", "args": {}}]}',
        '```json\n{"need_tools": false, "direct_reply": "你好！"}\n```',
        '{"need_tools": true, "tools": [{"name": "fund_query"}]}',  # 缺少 args
        'need_tools: true, tools: [{name: "schedule_query"}]',      # 非 JSON 格式
        '帮我查一下今天的账本',  # 纯自然语言
        '生成一张猫的图片',      # 纯自然语言
        '{"need_tools": true, "fund_query": [{"name": "fund_query", "args": {"code": "000001"}}]}',  # 异常结构
    ]

    print("=" * 60)
    print("JSON 提取测试")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test[:50]}...")
        result = extract_decision(test)
        print(f"  need_tools: {result['need_tools']}")
        print(f"  tools: {result['tools']}")
        if result['direct_reply']:
            print(f"  direct_reply: {result['direct_reply']}")

    print("\n" + "=" * 60)
    print("测试完成")