"""提取基金题到 基金问答智能体"""
import json, os

INPUT = r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json'
PROSPECTUS_FILE = r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl'
OUTPUT_DIR = r'C:\Users\freedom\Desktop\基金问答智能体'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'data', '基金问题.jsonl')

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 读所有题
with open(INPUT, 'r', encoding='utf-8') as f:
    all_q = [json.loads(l) for l in f if l.strip()]

# 读招股书ID
with open(PROSPECTUS_FILE, 'r', encoding='utf-8') as f:
    pros_ids = set(json.loads(l)['id'] for l in f if l.strip())

# 基金题 = 全部 - 招股书
fund_qs = [d for d in all_q if d['id'] not in pros_ids]

print(f"全部: {len(all_q)}")
print(f"招股书: {len(pros_ids)}")
print(f"基金: {len(fund_qs)}")

# 保存
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for d in fund_qs:
        f.write(json.dumps({"id": d["id"], "question": d["question"]}, ensure_ascii=False) + '\n')
print(f"已保存: {OUTPUT_FILE} ({len(fund_qs)} 条)")

# 展示基金题范围
print(f"\n基金题 ID 范围: {min(d['id'] for d in fund_qs)} ~ {max(d['id'] for d in fund_qs)}")
print(f"前3道:")
for d in fund_qs[:3]:
    print(f"  id={d['id']}: {d['question'][:80]}")
print(f"后3道:")
for d in fund_qs[-3:]:
    print(f"  id={d['id']}: {d['question'][:80]}")
