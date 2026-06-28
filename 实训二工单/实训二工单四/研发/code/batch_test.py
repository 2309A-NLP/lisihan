"""
批量测试脚本：快速两段式（生成SQL→执行→回答），不做ReAct循环
每50题保存一次结果，断点续跑
"""
import sys, os, json, time, sqlite3, re, requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))
Q_PATH = os.path.join(BASE, "..", "bs_challenge_financial_14b_dataset", "question_db.json")
OUT_DIR = os.path.join(BASE, "..", "output")
OUT_PATH = os.path.join(OUT_DIR, "batch_results.jsonl")
BATCH_SIZE = 50

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-jozgtgkyvzxikozrtkzgyfuptcamffjnpofushlitmktwyst"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"

DB_PATH = os.path.join(BASE, "..", "bs_challenge_financial_14b_dataset", "dataset", "博金杯比赛数据.db")

# 全局数据库连接（整个batch共用一个）
_db_conn = None
def get_db():
    global _db_conn
    if _db_conn is None:
        import sqlite3
        _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=60)
        _db_conn.execute("PRAGMA journal_mode=OFF")
        _db_conn.execute("PRAGMA synchronous=OFF")
        _db_conn.execute("PRAGMA cache_size=-80000")
    return _db_conn

def close_db():
    global _db_conn
    if _db_conn:
        _db_conn.close()
        _db_conn = None

SCHEMA = """
数据库10张表：

1. 基金基本信息: 基金代码,基金全称,基金简称,管理人,托管人,基金类型,成立日期(YYYYMMDD),到期日期,管理费率,托管费率

2. 基金股票持仓明细: 基金代码,基金简称,持仓日期(YYYYMMDD),股票代码,股票名称,数量,市值,市值占基金资产净值比,第N大重仓股,所在证券市场,所属国家(地区),报告类型

3. 基金债券持仓明细: 基金代码,基金简称,持仓日期(YYYYMMDD),债券类型,债券名称,持债数量,持债市值,持债市值占基金资产净值比,第N大重仓股,所在证券市场,所属国家(地区),报告类型

4. 基金可转债持仓明细: 基金代码,基金简称,持仓日期(YYYYMMDD),对应股票代码,债券名称,数量,市值,市值占基金资产净值比,第N大重仓股,所在证券市场,所属国家(地区),报告类型

5. 基金日行情表: 基金代码,交易日期(YYYYMMDD),单位净值,复权单位净值,累计单位净值,资产净值

6. A股票日行情表: 股票代码,交易日(YYYYMMDD),昨收盘(元),今开盘(元),最高价(元),最低价(元),收盘价(元),成交量(股),成交金额(元)

7. 港股票日行情表: 股票代码,交易日(YYYYMMDD),昨收盘(元),今开盘(元),最高价(元),最低价(元),收盘价(元),成交量(股),成交金额(元)

8. A股公司行业划分表: 股票代码,交易日期(YYYYMMDD),行业划分标准('中信行业分类'或'申万行业分类'),一级行业名称,二级行业名称

9. 基金规模变动表: 基金代码,基金简称,公告日期,截止日期,报告期期初基金总份额,报告期基金总申购份额,报告期基金总赎回份额,报告期期末基金总份额,定期报告所属年度,报告类型

10. 基金份额持有人结构: 基金代码,基金简称,公告日期,截止日期,机构投资者持有的基金份额,机构投资者持有的基金份额占总份额比例,个人投资者持有的基金份额,个人投资者持有的基金份额占总份额比例,定期报告所属年度,报告类型

重要：
- 日期YYYYMMDD无横线。涨跌幅=(收盘价-昨收盘)/昨收盘*100%
- 列名含中文括号如"收盘价(元)"，SQL要用双引号
- 表6用"交易日"，表5/8用"交易日期"
- 股票/基金代码是TEXT，要加引号
- JOIN优化：先查行业表缩小范围再JOIN（行业表1098万行很大），例如：
  WITH ind AS (SELECT 股票代码 FROM "A股公司行业划分表" WHERE 一级行业名称='XX' AND 交易日期<='YYYYMMDD')
  SELECT ... FROM "A股票日行情表" s JOIN ind ON s.股票代码=ind.股票代码 WHERE s.交易日='YYYYMMDD'
- 只查某天的数据时，先WHERE过滤日期再JOIN
- **日期过滤用 LIKE**：年份过滤 `成立日期 LIKE '2019%'`，不要用 strftime
- **示例**：查2019年成立的基金 → `SELECT count(*) FROM "基金基本信息" WHERE "管理人"='XXX' AND "成立日期" LIKE '2019%'`
"""

os.makedirs(OUT_DIR, exist_ok=True)

def call_llm(messages, temperature=0.01, max_tokens=1500):
    try:
        resp = requests.post(API_URL, json={
            "model": MODEL, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens
        }, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
"User-Agent": "Mozilla/5.0"}, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"【API错误{resp.status_code}】"
    except Exception as e:
        return f"【API异常: {e}】"

def exec_sql(sql):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        if not rows: return "无数据", cols
        return rows, cols
    except Exception as e:
        return None, str(e)

def quick_answer(question):
    """快速两段式：生成SQL → 执行 → 生成答案"""
    t0 = time.time()
    
    # 第1步：生成SQL
    sql_prompt = f"""{SCHEMA}

根据问题生成SQLite SQL查询。
重要规则：
- 问"数量"、"多少只"、"几个" → 用 COUNT(*)
- 问"股票代码"、"哪只" → 用 股票代码
- 问"涨跌幅超过X%" → 条件写成 (收盘价-昨收盘)/CAST(昨收盘 AS REAL)*100 > X
- 禁止 SELECT *，只选需要的列
只返回SQL，用```sql```代码块包裹。

问题：{question}"""
    
    sql_text = call_llm([
        {"role": "system", "content": "你是SQLite金融数据专家。只输出SQL，不要解释。"},
        {"role": "user", "content": sql_prompt}
    ], max_tokens=1000)
    
    if sql_text.startswith("【"):
        return sql_text, time.time() - t0
    
    # 提取SQL
    m = re.search(r'```sql\s*(.*?)\s*```', sql_text, re.DOTALL)
    sql = m.group(1).strip() if m else sql_text.strip()
    
    # 第2步：执行SQL
    result, meta = exec_sql(sql)
    if result is None:
        # SQL失败，重试一次更简单的SQL
        sql_text2 = call_llm([
            {"role": "system", "content": "你是SQLite金融专家。上次SQL错了，这次生成更简单的SQL，注意精确匹配列名。只输出SQL。"},
            {"role": "user", "content": f"问题：{question}\n数据库Schema：{SCHEMA}\n上次SQL错误：{meta}\n生成修复后的SQL："}
        ], max_tokens=1000)
        if not sql_text2.startswith("【"):
            m2 = re.search(r'```sql\s*(.*?)\s*```', sql_text2, re.DOTALL)
            sql2 = m2.group(1).strip() if m2 else sql_text2.strip()
            result, meta = exec_sql(sql2)
            if result is not None and result != "无数据":
                sql = sql2
    
    if result is None:
        return f"【SQL错误】{meta}", time.time() - t0
    if result == "无数据":
        # 无数据，重试一次（可能SQL缺了条件）
        sql_text2 = call_llm([
            {"role": "system", "content": "你是SQLite金融专家。上次SQL返回无数据，这次生成更准确的SQL。注意日期范围过滤。只输出SQL。"},
            {"role": "user", "content": f"问题：{question}\n数据库Schema：{SCHEMA}\n上次SQL：{sql}\n返回无数据，生成修复后的SQL："}
        ], max_tokens=1000)
        if not sql_text2.startswith("【"):
            m2 = re.search(r'```sql\s*(.*?)\s*```', sql_text2, re.DOTALL)
            sql2 = m2.group(1).strip() if m2 else sql_text2.strip()
            result, meta = exec_sql(sql2)
            if result is not None and result != "无数据":
                sql = sql2
    
    if result is None:
        return f"【SQL错误】{meta}", time.time() - t0
    if result == "无数据":
        return f"查询成功，无数据", time.time() - t0
    
    # 合理性校验：如果结果行数 > 500，很可能是SQL写错了（不该这么多）
    if len(result) > 500:
        # 重试一次，强调用COUNT和正确的JOIN
        retry_prompt = f"""{SCHEMA}

问题：{question}

上次SQL返回{len(result)}行，太多了！请重新生成SQL。
必须用 COUNT(*) 处理数量查询，确保用 WITH ind AS 先筛行业再JOIN。
只输出修正后的SQL，用```sql```包裹。"""
        sql_text3 = call_llm([
            {"role": "system", "content": "你SQL写错了，返回了大量数据。重新生成正确的SQL。"},
            {"role": "user", "content": retry_prompt}
        ], max_tokens=1000)
        if not sql_text3.startswith("【"):
            m3 = re.search(r'```sql\s*(.*?)\s*```', sql_text3, re.DOTALL)
            sql3 = m3.group(1).strip() if m3 else sql_text3.strip()
            result2, meta2 = exec_sql(sql3)
            if result2 is not None and result2 != "无数据" and len(result2) <= 500:
                result, meta, sql = result2, meta2, sql3
    
    # 第3步：生成答案
    result_str = f"列名: {meta}\n行数: {len(result)}\n"
    for r in result[:30]:
        result_str += f"  {r}\n"
    if len(result) > 30:
        result_str += f"  ...共{len(result)}行\n"
    
    answer = call_llm([
        {"role": "system", "content": "你是金融数据问答助手，根据查询结果用中文回答。答案要简洁准确。"},
        {"role": "user", "content": f"问题：{question}\n\n查询结果：\n{result_str}\n请给出简洁准确的答案。"}
    ], temperature=0.1, max_tokens=1000)
    
    elapsed = time.time() - t0
    return answer, elapsed


def load_questions():
    questions = []
    with open(Q_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): questions.append(json.loads(line))
    return questions

def load_existing():
    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results[r['id']] = r
    return results

def save_result(results_dict, qid):
    r = results_dict[qid]
    with open(OUT_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

def run_batch():
    questions = load_questions()
    existing = load_existing()
    total = len(questions)
    done = len(existing)
    
    print(f"共 {total} 题，已完成 {done} 题，待处理 {total - done} 题")
    
    if done > 0 and done < total:
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            for r in existing.values():
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    success = 0
    fail = 0
    batch_count = 0
    
    for i, q in enumerate(questions):
        qid = q['id'] if 'id' in q else i
        if qid in existing and existing[qid].get('answer'):
            r = existing[qid]
            if r.get('error'): fail += 1
            else: success += 1
            continue
        
        question_text = q.get('question', q.get('query', ''))
        if not question_text: continue
        
        print(f"\n[{i+1}/{total} 题#{qid}] {question_text[:40]}... ", end='', flush=True)
        
        answer, elapsed = quick_answer(question_text)
        
        is_error = (answer.startswith('【') or '错误' in answer or '失败' in answer)
        
        result = {
            'id': qid, 'question': question_text, 'answer': answer,
            'steps': 1, 'time': round(elapsed, 1),
            'error': 'error' if is_error else ''
        }
        existing[qid] = result
        save_result(existing, qid)
        
        if is_error:
            fail += 1
            print(f"✗ ({elapsed:.0f}s)")
        else:
            success += 1
            print(f"✓ ({elapsed:.0f}s)")
        
        batch_count += 1
        if batch_count >= BATCH_SIZE:
            print(f"\n{'='*50}")
            print(f"批次完成! 累计: 成功 {success}, 失败 {fail}, 总进度 {done + batch_count}/{total}")
            print(f"{'='*50}\n")
            batch_count = 0
    
    print(f"\n{'='*60}")
    print(f"  全部完成!  总计: {total}  成功: {success}  失败: {fail}")
    print(f"  成功率: {success/total*100:.1f}%")
    print(f"  输出: {OUT_PATH}")
    print(f"{'='*60}")
    close_db()

if __name__ == '__main__':
    run_batch()
