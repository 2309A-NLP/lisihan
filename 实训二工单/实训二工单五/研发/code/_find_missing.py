"""找基金分类中被误归的招股书题"""
import json

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    all_q = [json.loads(l) for l in f if l.strip()]

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    pros_ids = set(json.loads(l)['id'] for l in f if l.strip())

# 在基金分类中找：含公司名 + 问"什么/是谁/多少"的（像招股书问题）
import re
suspicious = []
for d in all_q:
    if d['id'] in pros_ids:
        continue
    q = d['question']
    # 有公司名
    has_company = bool(re.search(r'[\u4e00-\u9fff]{2,}(?:股份有限公司|有限公司)', q))
    # 问的是事实（什么、谁、多少），不是计算（涨跌幅、成交、收益率）
    is_fact_question = any(w in q for w in ['什么', '是谁', '是谁', '多少', '哪些', '哪个'])
    has_fund_calc = any(w in q for w in ['涨跌幅', '涨停', '跌停', '收盘价', '成交', '收益率', '净申购', '净赎回', '中证', '港股'])
    
    if has_company and is_fact_question and not has_fund_calc:
        suspicious.append(d)

print(f"基金分类中含公司名+事实问句（疑似招股书）: {len(suspicious)} 道")
print()
for d in suspicious:
    print(f"id={d['id']}: {d['question'][:120]}")
    print()
