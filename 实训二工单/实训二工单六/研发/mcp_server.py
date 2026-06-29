#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP 工具服务器 — 将5个工具暴露为 MCP Tool
工单编号：人工智能NLP-Agent数字人项目-智能体任务

功能：
1. 将5个后端服务封装为 MCP 工具
2. 支持 stdio 和 HTTP SSE 两种模式
3. 统一的错误处理和日志记录

运行方式：
    python mcp_server.py                    # stdio 模式（被 Agent 自动拉起）
    python mcp_server.py --http --port 8100 # HTTP SSE 模式（用于调试）
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional
from urllib.parse import urljoin

import requests
from mcp.server.fastmcp import FastMCP

# ============================================================
# 1. 配置
# ============================================================

# 5个后端服务的配置
TOOL_CONFIG = {
    "ledger": {
        "name": "记账本",
        "base_url": "http://127.0.0.1:8081",
        "endpoint": "/api/chat",
        "param_key": "message",
        "timeout": 30,
        "retry": 2,
    },
    "schedule": {
        "name": "日程提醒",
        "base_url": "http://127.0.0.1:5000",
        "endpoint": "/chat",
        "param_key": "message",
        "timeout": 30,
        "retry": 2,
    },
    "fund": {
        "name": "基金数据问答",
        "base_url": "http://127.0.0.1:5002",
        "endpoint": "/ask",
        "param_key": "question",
        "timeout": 30,
        "retry": 2,
    },
    "prospectus": {
        "name": "招股说明书问答",
        "base_url": "http://127.0.0.1:5003",
        "endpoint": "/ask",
        "param_key": "question",
        "timeout": 120,
        "retry": 1,
    },
    "image_gen": {
        "name": "文生图",
        "base_url": "http://127.0.0.1:7860",
        "endpoint": "/generate",
        "timeout": 120,  # 文生图需要更长时间
        "retry": 1,
    },
}

# ============================================================
# 2. 日志配置
# ============================================================

# 设置日志级别（可通过环境变量调整）
log_level = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MCPServer")

# 抑制第三方库的过多日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ============================================================
# 3. MCP 服务器初始化
# ============================================================

mcp = FastMCP(
    "智能管家Agent",
    log_level=log_level,
)

# ============================================================
# 4. 工具调用辅助函数
# ============================================================

def call_tool_api(
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = 30,
    retry: int = 2,
    param_key: str = "message",
) -> Dict[str, Any]:
    """
    调用后端服务的 HTTP API，支持重试

    Args:
        base_url: 服务基础 URL
        endpoint: API 端点
        payload: 请求体
        timeout: 超时时间（秒）
        retry: 重试次数
        param_key: 参数键名

    Returns:
        API 响应 JSON

    Raises:
        Exception: 所有重试失败后抛出
    """
    url = urljoin(base_url, endpoint)
    last_error = None

    for attempt in range(retry + 1):
        try:
            logger.debug(f"调用 {url} (尝试 {attempt + 1}/{retry + 1})")

            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"响应: {str(result)[:200]}...")
            return result

        except requests.exceptions.Timeout:
            last_error = f"请求超时 ({timeout}s)"
            logger.warning(f"尝试 {attempt + 1} 超时: {last_error}")

        except requests.exceptions.ConnectionError:
            last_error = f"连接失败: {base_url}"
            logger.warning(f"尝试 {attempt + 1} 连接失败: {last_error}")

        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP 错误: {e.response.status_code} - {e.response.text[:100]}"
            logger.warning(f"尝试 {attempt + 1} HTTP 错误: {last_error}")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"尝试 {attempt + 1} 异常: {last_error}")

        # 如果不是最后一次尝试，等待后重试
        if attempt < retry:
            wait_time = 2 ** attempt  # 指数退避：1, 2, 4...
            logger.debug(f"等待 {wait_time}s 后重试...")
            time.sleep(wait_time)

    raise Exception(last_error or "所有重试均失败")


def extract_reply(data: Dict[str, Any], param_key: str = "message") -> str:
    """
    从 API 响应中提取回复文本

    Args:
        data: API 响应 JSON
        param_key: 参数键名（用于日志）

    Returns:
        提取的回复文本
    """
    # 常见的回复字段
    reply_keys = ["reply", "answer", "text", "result", "message", "content"]

    for key in reply_keys:
        if key in data and data[key]:
            value = data[key]
            if isinstance(value, str):
                return value
            elif isinstance(value, (dict, list)):
                try:
                    return json.dumps(value, ensure_ascii=False)
                except:
                    return str(value)

    # 如果没有找到任何字段，返回整个响应
    return json.dumps(data, ensure_ascii=False)


# ============================================================
# 5. MCP 工具定义
# ============================================================

@mcp.tool(
    description="""📊 记账本 — 管理个人收支记录、查询历史账目、生成财务报表

【功能】
- 记录收入/支出：支持添加、修改、删除记账记录
- 查询账目：按日期、类别、成员筛选
- 统计分析：月度/年度收支汇总、分类统计
- 预算管理：设置和跟踪预算

【示例】
- "记录今天午餐花了35元"
- "这个月餐饮支出一共多少"
- "查一下上周的消费流水"
- "设置本月餐饮预算2000元"
""")
def ledger_query(query: str) -> str:
    """
    调用记账本工具

    Args:
        query: 用户的查询或操作指令

    Returns:
        记账本的处理结果
    """
    try:
        cfg = TOOL_CONFIG["ledger"]
        result = call_tool_api(
            base_url=cfg["base_url"],
            endpoint=cfg["endpoint"],
            payload={cfg["param_key"]: query},
            timeout=cfg["timeout"],
            retry=cfg["retry"],
            param_key=cfg["param_key"],
        )
        return extract_reply(result)
    except Exception as e:
        error_msg = f"⚠️ 记账本服务暂不可用：{e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool(
    description="""📅 日程提醒 — 管理日程事件、设置定时提醒、查询日程安排

【功能】
- 添加日程：创建新的日程事件
- 查询日程：按日期查看安排
- 修改/删除：更新或取消已有日程
- 提醒功能：设置定时提醒

【示例】
- "提醒我明天下午3点开会"
- "这周五有什么安排"
- "帮我加一个后天上午10点的牙医预约"
- "删除下周一2点的会议提醒"
""")
def schedule_query(query: str) -> str:
    """
    调用日程提醒工具

    Args:
        query: 用户的查询或操作指令

    Returns:
        日程提醒的处理结果
    """
    try:
        cfg = TOOL_CONFIG["schedule"]
        result = call_tool_api(
            base_url=cfg["base_url"],
            endpoint=cfg["endpoint"],
            payload={cfg["param_key"]: query},
            timeout=cfg["timeout"],
            retry=cfg["retry"],
            param_key=cfg["param_key"],
        )
        return extract_reply(result)
    except Exception as e:
        error_msg = f"⚠️ 日程提醒服务暂不可用：{e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool(
    description="""🎨 文生图 — 根据文本描述生成高质量图像

【功能】
- 文本生成图片：根据描述生成对应的图像
- 风格选择：支持多种艺术风格
- 图像尺寸：支持多种分辨率

【示例】
- "画一只可爱的橘猫坐在窗台上"
- "生成一张山水画风格的图片，有瀑布和松树"
- "帮我画一个未来城市的概念图"
- "绘制一幅星空下的浪漫场景"
""")
def image_generate(prompt: str) -> str:
    """
    生成图片，返回图片信息或URL

    Args:
        prompt: 图片描述文本

    Returns:
        图片生成结果信息
    """
    try:
        cfg = TOOL_CONFIG["image_gen"]
        url = urljoin(cfg["base_url"], cfg["endpoint"])

        logger.info(f"文生图请求: {prompt[:50]}...")

        # 尝试多种参数格式（兼容不同的后端）
        payload_attempts = [
            {"prompt": prompt, "action": "generate"},
            {"prompt": prompt},
            {"text": prompt, "action": "generate"},
            {"query": prompt},
        ]

        last_error = None
        for attempt, payload in enumerate(payload_attempts[:cfg.get("retry", 1) + 1]):
            try:
                response = requests.post(
                    url,
                    data=payload,
                    timeout=cfg["timeout"],
                )
                response.raise_for_status()
                data = response.json()

                # 检查是否成功
                if data.get("success") is False:
                    error_msg = data.get("error", data.get("message", "未知错误"))
                    last_error = error_msg
                    continue

                # 提取图片数据
                # 可能的返回格式：
                # 1. {"image": "base64_data", ...}
                # 2. {"image_path": "/path/to/image.png", ...}
                # 3. {"images": [{"url": "..."}], ...}
                # 4. {"result": "file.png", ...}
                if "image" in data:
                    # 返回 Base64 或 URL
                    image_data = data["image"]
                    if isinstance(image_data, str) and len(image_data) > 100:
                        # Base64 数据
                        return f"✅ 图片已生成！\n\n📷 图片数据已准备（{len(image_data)} 字符）\n提示：可在 Web UI 中查看"
                    else:
                        return f"✅ 图片已生成！\n\n📍 图片地址：{image_data}"

                elif "image_path" in data or "path" in data:
                    path = data.get("image_path") or data.get("path")
                    return f"✅ 图片已生成！\n\n📍 保存路径：{path}"

                elif "images" in data and data["images"]:
                    # 多图返回
                    images = data["images"]
                    if isinstance(images, list) and images:
                        if isinstance(images[0], dict) and "url" in images[0]:
                            urls = [img["url"] for img in images if "url" in img]
                            if urls:
                                return f"✅ 已生成 {len(urls)} 张图片\n\n📍 图片地址：\n" + "\n".join(f"  - {url}" for url in urls[:3])
                    elif isinstance(images[0], str):
                        return f"✅ 已生成 {len(images)} 张图片\n\n提示：可在 Web UI 中查看"

                elif "result" in data:
                    return f"✅ 图片已生成！\n\n📍 结果：{data['result']}"

                # 如果上面都没有，返回完整响应
                return f"✅ 图片生成完成\n\n📄 响应：{json.dumps(data, ensure_ascii=False)[:200]}..."

            except Exception as e:
                last_error = str(e)
                if attempt < cfg.get("retry", 1):
                    time.sleep(1)
                    continue

        return f"⚠️ 文生图服务返回错误：{last_error or '未知错误'}"

    except Exception as e:
        error_msg = f"⚠️ 文生图服务暂不可用：{e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool(
    description="""📈 基金数据问答 — 查询基金实时数据、历史统计、基金对比分析

【功能】
- 基金查询：按代码、名称查询基金信息
- 净值查询：最新净值、历史净值
- 收益率：近期收益率统计
- 基金对比：多只基金对比分析
- 市场行情：板块表现、热点追踪

【示例】
- "查一下000001基金的最新净值"
- "最近一个月哪些基金涨得最好"
- "比较一下沪深300ETF和中证500ETF的收益"
- "白酒板块的基金有哪些"
""")
def fund_query(question: str) -> str:
    """
    调用基金问答工具

    Args:
        question: 用户的问题

    Returns:
        基金问答的处理结果
    """
    try:
        cfg = TOOL_CONFIG["fund"]
        result = call_tool_api(
            base_url=cfg["base_url"],
            endpoint=cfg["endpoint"],
            payload={cfg["param_key"]: question},
            timeout=cfg["timeout"],
            retry=cfg["retry"],
            param_key=cfg["param_key"],
        )
        return extract_reply(result)
    except Exception as e:
        error_msg = f"⚠️ 基金问答服务暂不可用：{e}"
        logger.error(error_msg)
        return error_msg


@mcp.tool(
    description="""📄 招股说明书问答 — 解析招股说明书中的关键信息并回答问题

【功能】
- 财务数据：营收、利润、现金流等
- 公司信息：背景、股东、管理层
- 募投项目：资金用途、项目详情
- 发行信息：发行价格、数量、日期
- 风险因素：主要风险披露

【示例】
- "招股书中公司去年的营收是多少"
- "这家公司的募投项目有哪些"
- "招股说明书里的毛利率是多少"
- "公司的股权结构是什么样的"
- "发行价格是多少"
""")
def prospectus_query(question: str) -> str:
    """
    调用招股书问答工具

    Args:
        question: 用户的问题

    Returns:
        招股书问答的处理结果
    """
    try:
        cfg = TOOL_CONFIG["prospectus"]
        result = call_tool_api(
            base_url=cfg["base_url"],
            endpoint=cfg["endpoint"],
            payload={cfg["param_key"]: question},
            timeout=cfg["timeout"],
            retry=cfg["retry"],
            param_key=cfg["param_key"],
        )
        return extract_reply(result)
    except Exception as e:
        error_msg = f"⚠️ 招股书问答服务暂不可用：{e}"
        logger.error(error_msg)
        return error_msg


# ============================================================
# 6. 健康检查（可选）
# ============================================================

@mcp.resource("health://status")
def health_status() -> str:
    """获取服务器健康状态"""
    statuses = []
    for name, cfg in TOOL_CONFIG.items():
        try:
            # 尝试连接服务
            url = urljoin(cfg["base_url"], cfg.get("endpoint", ""))
            response = requests.get(cfg["base_url"], timeout=3)
            if response.status_code < 500:
                statuses.append(f"✅ {cfg['name']}: 正常运行")
            else:
                statuses.append(f"⚠️ {cfg['name']}: 响应异常 ({response.status_code})")
        except Exception:
            statuses.append(f"❌ {cfg['name']}: 不可达")

    return "MCP 服务器状态\n" + "=" * 30 + "\n" + "\n".join(statuses)


# ============================================================
# 7. 主入口
# ============================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="MCP 工具服务器 — 智能管家 5 合 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式：
  stdio 模式（默认）：python mcp_server.py
    用于 Agent 子进程调用，通过标准输入输出通信
  
  HTTP SSE 模式：python mcp_server.py --http --port 8100
    用于调试和测试，可通过浏览器访问

环境变量：
  MCP_LOG_LEVEL: 日志级别 (DEBUG/INFO/WARNING/ERROR)
  示例: set MCP_LOG_LEVEL=DEBUG && python mcp_server.py
        """
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="以 HTTP SSE 模式运行（默认为 stdio 模式）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="HTTP 模式监听的端口（默认: 8100）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查所有后端服务状态并退出"
    )

    args = parser.parse_args()

    # 检查模式
    if args.check:
        print("检查后端服务状态...")
        print("=" * 50)
        all_ok = True
        for name, cfg in TOOL_CONFIG.items():
            try:
                response = requests.get(cfg["base_url"], timeout=3)
                if response.status_code < 500:
                    print(f"✅ {cfg['name']}: 正常运行 (端口 {cfg['base_url'].split(':')[-1]})")
                else:
                    print(f"⚠️ {cfg['name']}: 响应异常 (HTTP {response.status_code})")
                    all_ok = False
            except requests.exceptions.ConnectionError:
                print(f"❌ {cfg['name']}: 服务未启动或不可达")
                all_ok = False
            except Exception as e:
                print(f"❌ {cfg['name']}: 检查失败 - {e}")
                all_ok = False
        print("=" * 50)
        sys.exit(0 if all_ok else 1)

    # 启动 MCP 服务器
    logger.info(f"启动 MCP 服务器 (模式: {'HTTP SSE' if args.http else 'stdio'})")
    if args.http:
        logger.info(f"监听端口: {args.port}")
        logger.info(f"访问地址: http://127.0.0.1:{args.port}")

    try:
        if args.http:
            mcp.run(transport="sse", port=args.port)
        else:
            mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()