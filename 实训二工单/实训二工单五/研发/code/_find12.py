"""找那12道漏的题：基金分类中无基金特征的问题"""
import json, re

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    all_q = {d['id']: d['question'] for d in [json.loads(l) for l in f if l.strip()]}

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    pros_ids = set(json.loads(l)['id'] for l in f if l.strip())

# 基金分类中的题
fund_qs = {qid: q for qid, q in all_q.items() if qid not in pros_ids}

# 去掉明显的基金题（含基金、股票、涨跌幅等关键词）
FUND_KEYWORDS = ['基金', '涨跌幅', '涨停', '跌停', '收盘价', '开盘价', '成交', 
                 '净申购', '净赎回', '中证', '港股', '代码', '收益率', '净值',
                 '重仓', '债券', '可转债', '持有份额', '持有人', '规模变动',
                 '入股', '日收益率', '股票数量']

not_obvious_fund = []
for qid, q in fund_qs.items():
    # 过滤掉明显的基金题
    has_fund_kw = any(kw in q for kw in FUND_KEYWORDS)
    if not has_fund_kw:
        not_obvious_fund.append((qid, q))

print(f"基金分类共 {len(fund_qs)} 道")
print(f"其中不含明显基金关键词的: {len(not_obvious_fund)} 道")
print()

for qid, q in sorted(not_obvious_fund):
    # 看看是不是带公司名的
    has_company = bool(re.search(r'[\u4e00-\u9fff]{2,}(?:股份有限公司|有限公司)', q))
    tag = " [有公司名]" if has_company else ""
    print(f"id={qid}{tag}: {q[:120]}")
    print()
