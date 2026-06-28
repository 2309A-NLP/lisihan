"""核对用户担心的12道 + 找剩下的漏网之鱼"""
import json, re

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    all_q = {d['id']: d['question'] for d in [json.loads(l) for l in f if l.strip()]}

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    pros_ids = set(json.loads(l)['id'] for l in f if l.strip())

# 1. 用户指出的12道
print("=== 用户担心的12道题 ===")
user_ids = [21, 40, 46, 47, 59, 85, 175, 249, 502, 660, 800, 543]
for qid in user_ids:
    status = "已在内" if qid in pros_ids else "被归为基金"
    print(f"id={qid} {status}: {all_q.get(qid, '')[:100]}")

# 2. 基金分类中检查华润微电子、华塑股份等
print("\n=== 基金分类中疑似漏网的招股书题 ===")
fund_qs = {qid: q for qid, q in all_q.items() if qid not in pros_ids}
check_companies = ['华润微电子', '华塑股份', '万邦生化', '华瑞', '君正', '联化', '森赫', '信立泰', '汉嘉', '天宜上佳', '兴图新科']
for qid, q in sorted(fund_qs.items()):
    for c in check_companies:
        if c in q:
            print(f"id={qid} [含'{c}']: {q[:120]}")
            break
