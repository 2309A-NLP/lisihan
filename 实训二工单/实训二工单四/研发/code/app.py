# -*- coding: utf-8 -*-

"""
基金数据问答智能体 - ReAct Agent (Web服务)
AI自主决策：查数据库、搜招股书、多步推理
支持 SQLite 数据库查询，准确回答统计数据问题

工作流程：
1. 用户提出问题 → 2. 优先使用 ReAct Agent (LLM生成SQL)
3. 如果失败则降级到备用解析器 → 4. 返回查询结果
"""
import sys
import os
import json
import threading
import sqlite3
import re
from datetime import datetime
from pathlib import Path

# 注意：不在此处修改 sys.stdout，避免 Windows 编码冲突
# 如需解决编码问题，请在启动脚本中设置：set PYTHONIOENCODING=utf-8

sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, request, render_template, jsonify
from react_agent import react_agent

app = Flask(__name__)

# ============================================================
# 数据库配置
# ============================================================

# 数据库路径（指向比赛数据文件）
DB_PATH = r"C:\Users\freedom\Desktop\agent\基金问答智能体\bs_challenge_financial_14b_dataset\dataset\博金杯比赛数据.db"

# 行业名称映射（将用户输入的不同表述映射到数据库中的标准名称）
INDUSTRY_MAP = {
    "建筑材料": "建筑材料",
    "建筑": "建筑材料",
    "建材": "建筑材料",
    "综合金融": "综合金融",
    "金融": "综合金融",
    # 可根据需要添加更多行业映射
}


def get_db_connection():
    """获取数据库连接，如果文件不存在则返回 None"""
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)


# ============================================================
# 备用解析器（当 ReAct Agent 不可用时使用）
# ============================================================

def parse_question(question: str) -> dict:
    """
    解析用户问题，提取关键信息（备用解析器）

    注意：这是备选方案，主要查询逻辑已交由 react_agent 处理
    此函数仅用于简单场景的快速匹配

    Args:
        question: 用户问题

    Returns:
        包含 date, industry, threshold, query_type 的字典
    """
    result = {
        "date": None,  # 日期 YYYYMMDD
        "industry": None,  # 行业名称
        "condition": ">",  # 比较条件（默认大于）
        "threshold": 5,  # 涨跌幅阈值（默认 5%）
        "query_type": "count",  # 查询类型：count(统计) / list(列表)
    }

    # ----- 1. 提取日期 -----
    # 支持格式：20210415 / 2021-04-15 / 2021年4月15日
    date_patterns = [
        r'(\d{4})(\d{2})(\d{2})',  # 20210415
        r'(\d{4})[-/](\d{2})[-/](\d{2})',  # 2021-04-15 或 2021/04/15
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # 2021年4月15日
    ]

    for pattern in date_patterns:
        match = re.search(pattern, question)
        if match:
            if len(match.groups()) == 3:
                y, m, d = match.groups()
                result["date"] = f"{y}{int(m):02d}{int(d):02d}"
            break

    # ----- 2. 提取行业 -----
    industry_patterns = [
        r'(\S+?)行业',  # 建筑材料行业
        r'(\S+?)板块',  # 建筑材料板块
        r'一级行业[是为]?\s*([^\s,，]+)',  # 一级行业为建筑材料
        r'([^\s,，]+)一级行业',  # 建筑材料一级行业
    ]

    for pattern in industry_patterns:
        match = re.search(pattern, question)
        if match:
            industry_name = match.group(1).strip()
            # 尝试映射到标准名称
            for key, value in INDUSTRY_MAP.items():
                if key in industry_name or industry_name in key:
                    result["industry"] = value
                    break
            if not result["industry"]:
                result["industry"] = industry_name
            break

    # ----- 3. 提取涨幅条件 -----
    threshold_patterns = [
        r'涨幅[超过高于大于]{2,}\s*(\d+(?:\.\d+)?)%',  # 涨幅超过5%
        r'超过\s*(\d+(?:\.\d+)?)%',  # 超过5%
        r'大于\s*(\d+(?:\.\d+)?)%',  # 大于5%
        r'高于\s*(\d+(?:\.\d+)?)%',  # 高于5%
        r'[>＞]\s*(\d+(?:\.\d+)?)%',  # >5%
    ]

    for pattern in threshold_patterns:
        match = re.search(pattern, question)
        if match:
            result["threshold"] = float(match.group(1))
            break

    # ----- 4. 判断查询类型 -----
    if "数量" in question or "多少" in question or "几个" in question or "多少只" in question:
        result["query_type"] = "count"  # 统计数量
    elif "列出" in question or "哪些" in question or "分别" in question:
        result["query_type"] = "list"  # 列出列表
    else:
        result["query_type"] = "count"  # 默认统计数量

    return result


def query_stock_count(date: str, industry: str, threshold: float, condition: str = ">") -> int:
    """
    查询满足条件的股票数量（备用查询函数）

    Args:
        date: 日期 (YYYYMMDD)
        industry: 行业名称
        threshold: 涨跌幅阈值
        condition: 比较条件 (>, >=, <, <=)

    Returns:
        股票数量，-1 表示查询失败
    """
    conn = get_db_connection()
    if not conn:
        return -1

    try:
        cursor = conn.cursor()

        # 涨跌幅 = (收盘价 - 昨收盘) / 昨收盘 * 100
        query = f'''
        SELECT COUNT(DISTINCT h."股票代码")
        FROM "A股票日行情表" h
        INNER JOIN "A股公司行业划分表" i 
            ON h."股票代码" = i."股票代码"
        WHERE h."交易日" = ?
          AND i."一级行业名称" = ?
          AND (h."收盘价(元)" - h."昨收盘(元)") / h."昨收盘(元)" * 100 {condition} ?
        '''

        cursor.execute(query, (date, industry, threshold))
        result = cursor.fetchone()[0]
        conn.close()
        return result

    except Exception as e:
        print(f"[DB Error] {e}")
        conn.close()
        return -1


def query_stock_list(date: str, industry: str, threshold: float, condition: str = ">") -> list:
    """
    查询满足条件的股票列表（备用查询函数）

    Args:
        date: 日期 (YYYYMMDD)
        industry: 行业名称
        threshold: 涨跌幅阈值
        condition: 比较条件

    Returns:
        股票列表 [(code, change_pct), ...]
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()

        query = f'''
        SELECT 
            h."股票代码",
            (h."收盘价(元)" - h."昨收盘(元)") / h."昨收盘(元)" * 100 as 涨跌幅
        FROM "A股票日行情表" h
        INNER JOIN "A股公司行业划分表" i 
            ON h."股票代码" = i."股票代码"
        WHERE h."交易日" = ?
          AND i."一级行业名称" = ?
          AND (h."收盘价(元)" - h."昨收盘(元)") / h."昨收盘(元)" * 100 {condition} ?
        ORDER BY 涨跌幅 DESC
        '''

        cursor.execute(query, (date, industry, threshold))
        rows = cursor.fetchall()
        conn.close()
        return [(row[0], row[1]) for row in rows]

    except Exception as e:
        print(f"[DB Error] {e}")
        conn.close()
        return []


def format_date(date_str: str) -> str:
    """将 YYYYMMDD 格式化为 YYYY年MM月DD日 的友好显示格式"""
    if len(date_str) == 8:
        return f"{date_str[:4]}年{int(date_str[4:6])}月{int(date_str[6:8])}日"
    return date_str


# ============================================================
# 主问答函数
# ============================================================

def answer_question(question: str) -> tuple:
    """
    回答用户问题，优先使用 ReAct Agent，失败时降级到备用方案

    工作流程：
    1. 首先尝试使用 react_agent（LLM 自主生成 SQL，最灵活）
    2. 如果 react_agent 失败，降级到备用解析器（快速匹配）

    Args:
        question: 用户问题

    Returns:
        (answer, error) 元组
    """
    # ===== 策略1：优先使用 ReAct Agent =====
    # ReAct Agent 会自主理解问题、生成 SQL、查询数据库
    # 这是最灵活的方式，可以处理各种复杂问题
    try:
        print(f"[Agent] 使用 ReAct Agent 处理问题: {question[:50]}...")
        answer, steps = react_agent(question)
        # 如果返回的答案不是错误信息，直接返回
        if answer and not answer.startswith('【'):
            return answer, None
        print(f"[Agent] ReAct Agent 返回错误，降级到备用方案")
    except Exception as e:
        print(f"[Agent] ReAct Agent 异常: {e}，降级到备用方案")

    # ===== 策略2：备用解析器 =====
    # 适用于简单场景的快速匹配，不依赖 LLM
    print(f"[Parser] 使用备用解析器")
    parsed = parse_question(question)

    # 如果解析到了日期和行业，尝试数据库查询
    if parsed["date"] and parsed["industry"]:
        date = parsed["date"]
        industry = parsed["industry"]
        threshold = parsed["threshold"]
        query_type = parsed["query_type"]

        print(f"[Parser] 日期: {date}, 行业: {industry}, 阈值: {threshold}%, 类型: {query_type}")

        # 查询数据库
        count = query_stock_count(date, industry, threshold)

        if count >= 0:
            date_formatted = format_date(date)
            industry_name = industry

            if query_type == "list":
                # 列出具体股票
                stocks = query_stock_list(date, industry, threshold)
                if stocks:
                    stock_list = "\n".join([f"  - {code}: {change:.2f}%" for code, change in stocks])
                    answer = f"{date_formatted}，{industry_name}一级行业涨幅超过{threshold}%的股票共有 {count} 只：\n{stock_list}"
                else:
                    answer = f"{date_formatted}，{industry_name}一级行业涨幅超过{threshold}%的股票数量为 {count} 只。"
            else:
                # 默认统计数量
                answer = f"{date_formatted}，{industry_name}一级行业涨幅超过{threshold}%的股票数量为 {count} 只。"

            return answer, None
        else:
            print("[Parser] 数据库查询失败")

    # ===== 策略3：所有方法都失败 =====
    return "抱歉，我无法理解您的问题。请尝试使用更明确的描述，例如：'20210415日，建筑材料一级行业涨幅超过5%的股票数量'", None


# ============================================================
# Flask 路由
# ============================================================

@app.route('/')
def index():
    """首页：渲染问答界面"""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """处理问答请求：接收问题，返回答案"""
    data = request.get_json()
    question = (data.get('question') or data.get('query') or '').strip()
    if not question:
        return jsonify({'answer': '请输入问题', 'type': ''})

    try:
        answer, error = answer_question(question)
        is_error = answer.startswith('【') or error is not None

        return jsonify({
            'answer': answer,
            'type': 'AI',
            'error': 'error' if is_error else ''
        })
    except Exception as e:
        return jsonify({
            'answer': f'出错: {str(e)}',
            'type': '',
            'error': str(e)
        })


# ============================================================
# 启动入口
# ============================================================

if __name__ == '__main__':
    import webbrowser

    port = 5002

    # ===== 打印启动信息 =====
    # 注意：如果控制台无法显示 [OK] 等字符，请在启动前设置：
    # set PYTHONIOENCODING=utf-8
    print("=" * 50)
    print("  基金数据问答智能体 v2.0 (支持数据库查询)")
    print("=" * 50)

    # 检查数据库是否存在
    if os.path.exists(DB_PATH):
        print(f"[OK] 数据库已连接: {DB_PATH}")
        # 测试连接
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM 'A股票日行情表'")
            count = cursor.fetchone()[0]
            print(f"[OK] 数据加载成功: {count} 条行情记录")
            conn.close()
        except Exception as e:
            print(f"[WARN] 数据库连接测试失败: {e}")
    else:
        print(f"[WARN] 数据库未找到: {DB_PATH}")
        print("   AI 模式将作为备选方案")

    print("=" * 50)
    print(f"[OK] 启动中... http://localhost:{port}")
    print("=" * 50)

    # 自动打开浏览器
    threading.Timer(0.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(host='localhost', port=port, debug=False)