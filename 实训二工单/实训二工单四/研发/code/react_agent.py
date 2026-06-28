# -*- coding: utf-8 -*-

"""
ReAct 智能体 - 基金数据问答
AI 自主决定查什么SQL，多步推理给出答案

工作流程：
1. 用户提问 → 2. LLM 思考需要什么数据 → 3. 生成 SQL → 4. 执行查询 → 5. 返回结果
支持：股票行情查询、行业分类查询、涨跌幅计算、基金信息查询等
"""

import sys  # ⚠️ 必须最先导入，否则下面的 sys.platform 会报错
import json
import re
import time
import sqlite3
import os
import requests

# ===== 修复 Windows 控制台编码问题 =====
# 解决 Windows 下 print 无法显示 Unicode 字符（如 ✅）的问题
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================
# 1. API 配置
# ============================================================

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-jozgtgkyvzxikozrtkzgyfuptcamffjnpofushlitmktwyst"
MODEL = "deepseek-ai/DeepSeek-V3"

# ============================================================
# 2. 数据库配置
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "..", "bs_challenge_financial_14b_dataset", "dataset", "博金杯比赛数据.db")

# ============================================================
# 3. LLM 调用函数
# ============================================================

def call_llm(messages, temperature=0.01, max_tokens=2000):
    """
    调用 SiliconFlow API

    Args:
        messages: 对话消息列表
        temperature: 温度参数（0-1），越低越确定
        max_tokens: 最大输出 token 数

    Returns:
        LLM 返回的文本内容
    """
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            # 失败后等待并重试（指数退避：3s, 6s, 9s）
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            print(f"[LLM] 调用失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(3)
    return "【LLM调用失败】"


# ============================================================
# 4. 数据库连接和查询
# ============================================================

_db_conn = None

def get_db():
    """
    获取数据库连接（单例模式）
    使用 PRAGMA 优化查询性能
    """
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        _db_conn.execute("PRAGMA journal_mode=OFF")      # 关闭日志，提高写入性能
        _db_conn.execute("PRAGMA synchronous=OFF")       # 关闭同步，提高速度
        _db_conn.execute("PRAGMA cache_size=10000")      # 增加缓存大小
    return _db_conn


def query_db(sql):
    """
    执行 SQL 查询并返回格式化的结果

    Args:
        sql: SQL 查询语句

    Returns:
        格式化的查询结果字符串，便于 LLM 理解
    """
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []

        if not rows:
            return "无数据"

        # 格式化输出，便于 LLM 理解
        text = f"列名: {cols}\n行数: {len(rows)}\n"
        for r in rows[:30]:
            text += f"  {r}\n"
        if len(rows) > 30:
            text += f"  ...共{len(rows)}行\n"
        return text
    except Exception as e:
        return f"SQL错误: {e}"


# ============================================================
# 5. 数据库 Schema（供 LLM 参考生成 SQL）
# ============================================================

SCHEMA = """
数据库10张表：

1. 基金基本信息: 基金代码,基金全称,基金简称,管理人,托管人,基金类型,成立日期(YYYYMMDD),到期日期,管理费率,托管费率
2. 基金股票持仓明细: 基金代码,基金简称,持仓日期(YYYYMMDD),股票代码,股票名称,数量,市值,市值占基金资产净值比,第N大重仓股,报告类型
3. 基金债券持仓明细: 基金代码,基金简称,持仓日期(YYYYMMDD),债券类型,债券名称,持债数量,持债市值,持债市值占基金资产净值比,第N大重仓股,报告类型
4. 基金可转债持仓明细: 基金代码,基金简称,持仓日期(YYYYMMDD),对应股票代码,债券名称,数量,市值,市值占基金资产净值比,第N大重仓股,报告类型
5. 基金日行情表: 基金代码,交易日期(YYYYMMDD),单位净值,复权单位净值,累计单位净值,资产净值
6. A股票日行情表: 股票代码,交易日(YYYYMMDD),昨收盘(元),今开盘(元),最高价(元),最低价(元),收盘价(元),成交量(股),成交金额(元)
7. 港股票日行情表: 股票代码,交易日(YYYYMMDD),昨收盘(元),今开盘(元),最高价(元),最低价(元),收盘价(元),成交量(股),成交金额(元)
8. A股公司行业划分表: 股票代码,交易日期(YYYYMMDD),行业划分标准('中信行业分类' 或 '申万行业分类'),一级行业名称,二级行业名称
9. 基金规模变动表: 基金代码,基金简称,公告日期,截止日期,报告期期初基金总份额,报告期基金总申购份额,报告期基金总赎回份额,报告期期末基金总份额,定期报告所属年度,报告类型
10. 基金份额持有人结构: 基金代码,基金简称,公告日期,截止日期,机构投资者持有的基金份额,机构投资者持有的基金份额占总份额比例,个人投资者持有的基金份额,个人投资者持有的基金份额占总份额比例,定期报告所属年度,报告类型

【重要提示】
- 日期格式：YYYYMMDD（无横线），如 20210415
- 涨跌幅公式：(收盘价 - 昨收盘) / 昨收盘 * 100
- 列名含中文括号（如"收盘价(元)"）必须用双引号包裹，例如: "收盘价(元)"
- 股票代码和基金代码是 TEXT 类型，查询时加引号，例如: '600120'
- 表6使用"交易日"，表5/8使用"交易日期"
- 年份过滤用 LIKE '2019%'，不要用 strftime
- ⚠️ 行业划分标准字段值为 '中信行业分类' 或 '申万行业分类'（必须是完整名称！）
"""


# ============================================================
# 6. System Prompt（核心指令 - 指导 LLM 如何生成 SQL）
# ============================================================

SYSTEM_PROMPT = """你是一个基金数据智能体。你只有一个工具：

### 工具: query_db
参数：SQL语句。执行SQL查询基金数据库，返回结果。

### 工作流程：
1. 理解用户问题
2. 思考需要查询哪些数据
3. 生成 SQL 语句
4. 执行查询
5. 根据查询结果给出答案

### 输出格式：
需要查数据时：
Thought: 我理解用户需要...，需要查询...
Action: query_db
Action Input: 完整的 SQL 查询语句

有答案时（直接输出最终答案）：
Final Answer: 简洁的答案（用中文）

### 规则：
- 一次只能调用一个工具
- SQL列名含中文括号要用双引号包裹
- 日期用 YYYYMMDD 格式（无横线）
- 先缩小范围再 JOIN（使用 CTE 或子查询）
- 最终答案简洁准确，用中文
- ⚠️ 行业划分标准必须用完整名称：'中信行业分类' 或 '申万行业分类'（不要用 '中信' 或 '申万'）

### SQL 示例（仅作参考，根据实际问题调整）：

**示例1：查询某行业涨幅超过5%的股票数量**
问题：20210415日，建筑材料一级行业涨幅超过5%的股票数量
SQL：
WITH ind AS (
    SELECT DISTINCT 股票代码 
    FROM "A股公司行业划分表" 
    WHERE 一级行业名称='建筑材料' 
      AND 交易日期<='20210415'
)
SELECT COUNT(*) 
FROM "A股票日行情表" s 
JOIN ind ON s.股票代码=ind.股票代码 
WHERE s.交易日='20210415' 
  AND (s."收盘价(元)"-s."昨收盘(元)")/s."昨收盘(元)"*100 > 5

**示例2：查询某行业（中信分类）涨幅最大的股票代码和涨跌幅**
问题：20210105日，中信行业分类的综合金融行业中，涨跌幅最大的股票代码和涨跌幅
SQL：
WITH ind AS (
    SELECT DISTINCT 股票代码 
    FROM "A股公司行业划分表" 
    WHERE 一级行业名称='综合金融' 
      AND 行业划分标准='中信行业分类'
      AND 交易日期<='20210105'
)
SELECT s.股票代码, 
       (s."收盘价(元)"-s."昨收盘(元)")/s."昨收盘(元)"*100 AS 涨跌幅
FROM "A股票日行情表" s 
JOIN ind ON s.股票代码=ind.股票代码 
WHERE s.交易日='20210105' 
ORDER BY 涨跌幅 DESC 
LIMIT 1

**示例3：查询某行业（申万分类）涨幅超过5%的所有股票**
问题：列出20210415日申万行业分类的建筑材料一级行业涨幅超过5%的股票
SQL：
WITH ind AS (
    SELECT DISTINCT 股票代码 
    FROM "A股公司行业划分表" 
    WHERE 一级行业名称='建筑材料' 
      AND 行业划分标准='申万行业分类'
      AND 交易日期<='20210415'
)
SELECT s.股票代码, 
       (s."收盘价(元)"-s."昨收盘(元)")/s."昨收盘(元)"*100 AS 涨跌幅
FROM "A股票日行情表" s 
JOIN ind ON s.股票代码=ind.股票代码 
WHERE s.交易日='20210415' 
  AND (s."收盘价(元)"-s."昨收盘(元)")/s."昨收盘(元)"*100 > 5
ORDER BY 涨跌幅 DESC

**示例4：查询某基金的管理人**
问题：基金代码为000001的基金管理人是谁
SQL：
SELECT 管理人 FROM "基金基本信息" WHERE 基金代码='000001'

**示例5：查询2019年成立的基金数量**
问题：2019年成立的基金有多少只
SQL：
SELECT COUNT(*) FROM "基金基本信息" WHERE 成立日期 LIKE '2019%'

**示例6：查询某行业所有股票及涨跌幅**
问题：20210415日建筑材料一级行业所有股票的涨跌幅
SQL：
WITH ind AS (
    SELECT DISTINCT 股票代码 
    FROM "A股公司行业划分表" 
    WHERE 一级行业名称='建筑材料' 
      AND 交易日期<='20210415'
)
SELECT s.股票代码, 
       (s."收盘价(元)"-s."昨收盘(元)")/s."昨收盘(元)"*100 AS 涨跌幅
FROM "A股票日行情表" s 
JOIN ind ON s.股票代码=ind.股票代码 
WHERE s.交易日='20210415' 
ORDER BY 涨跌幅 DESC

### 数据库Schema：
""" + SCHEMA


# ============================================================
# 7. 辅助函数：提取内容块
# ============================================================

def extract_block(text, tag):
    """
    从 LLM 输出中提取指定标签的内容

    Args:
        text: LLM 输出文本
        tag: 标签名（如 Action, Action Input, Final Answer）

    Returns:
        提取的内容，未找到返回 None
    """
    # 匹配模式：标签: 内容（直到遇到下一个标签或结束）
    pattern = rf'{tag}:\s*(.*?)(?=\n(?:Thought|Action|Final Answer)|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


# ============================================================
# 8. ReAct Agent 主函数
# ============================================================

def react_agent(question, max_steps=8):
    """
    ReAct (Reasoning + Acting) 智能体主函数

    工作流程：
    1. 将用户问题 + System Prompt 发送给 LLM
    2. LLM 决定是否需要查询数据库
    3. 如果需要，生成 SQL 并执行
    4. 将查询结果反馈给 LLM
    5. LLM 根据结果生成最终答案

    Args:
        question: 用户问题
        max_steps: 最大推理步数（防止死循环）

    Returns:
        (final_answer, steps_list) 元组
    """
    # 初始化对话消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}"}
    ]
    steps = []

    for step in range(max_steps):
        # 1. 调用 LLM 进行推理
        print(f"[Agent] 步骤 {step+1}/{max_steps}: 调用 LLM...")
        response = call_llm(messages, temperature=0.01, max_tokens=1500)
        messages.append({"role": "assistant", "content": response})

        # 2. 检查是否已有最终答案
        final = extract_block(response, "Final Answer")
        if final:
            print(f"[Agent] ✅ 获得最终答案")
            return final, steps

        # 3. 提取 Action 和 Action Input
        action = extract_block(response, "Action")
        act_input = extract_block(response, "Action Input")

        # 如果格式不对，直接返回 LLM 原始输出
        if not action or not act_input:
            print(f"[Agent] ⚠️ 无法解析 Action，返回原始响应")
            return response, steps

        # 4. 记录步骤
        steps.append({
            "action": action,
            "input": act_input[:80] + ("..." if len(act_input) > 80 else "")
        })
        print(f"[Agent] Action: {action}")
        print(f"[Agent] Input: {act_input[:100]}...")

        # 5. 执行工具
        if action == "query_db":
            result = query_db(act_input)
        else:
            result = f"未知工具: {action}"

        print(f"[Agent] Observation: {result[:200]}...")

        # 6. 将观察结果加入对话，供 LLM 下一步推理
        messages.append({"role": "user", "content": f"Observation: {result}"})

    return "【已达最大推理步数，无法给出答案】", steps


# ============================================================
# 9. 测试入口
# ============================================================

if __name__ == "__main__":
    # 测试问题列表
    test_questions = [
        "20210415日，建筑材料一级行业涨幅超过5%的股票数量",
        "20210105日，中信行业分类的综合金融行业中，涨跌幅最大的股票代码和涨跌幅",
        "列出20210415日建筑材料一级行业涨幅超过5%的股票",
        "20210415日申万行业分类的建筑材料一级行业涨幅超过5%的股票有哪些",
    ]

    print("=" * 60)
    print("  ReAct Agent 基金数据问答测试")
    print("=" * 60)

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        print("-" * 40)
        answer, steps = react_agent(q)
        print(f"\nA: {answer}")
        print(f"Steps: {len(steps)}")

    print("\n" + "=" * 60)
    print("测试完成")