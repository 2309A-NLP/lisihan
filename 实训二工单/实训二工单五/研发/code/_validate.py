"""验证分类质量"""
import json

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    prospectus = [json.loads(l) for l in f if l.strip()]

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    all_q = [json.loads(l) for l in f if l.strip()]

prospectus_ids = set(d['id'] for d in prospectus)
fund_ids = [d for d in all_q if d['id'] not in prospectus_ids]

print(f"总: {len(all_q)}, 招股书: {len(prospectus_ids)}, 基金: {len(fund_ids)}")

# 1. 被归为招股书 but 包含基金关键词（可能误分）
print("\n=== 招股书分类中含基金关键词（可能误分）===")
for d in prospectus:
    q = d['question']
    kw = [w for w in ['基金','净值','收盘价','涨跌幅','成交','行业','重仓'] if w in q]
    if kw:
        print(f"  id={d['id']} [{','.join(kw)}]: {q[:80]}")

# 2. 被归为基金 but 可能是招股书
print("\n=== 基金分类中可能是招股书（含公司名但无基金词）===")
for d in fund_ids:
    q = d['question']
    if ('股份有限公司' in q or '有限公司' in q) and '基金' not in q and '股票' not in q and '代码' not in q:
        print(f"  id={d['id']}: {q[:100]}")

# 3. 展示问题范围
print(f"\n招股书问题 ID 范围: {min(prospectus_ids)} ~ {max(prospectus_ids)}")
print(f"基金问题 ID 范围: {min(d['id'] for d in fund_ids)} ~ {max(d['id'] for d in fund_ids)}")
