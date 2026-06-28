"""验证基金文件里没有那12道题"""
import json

fund_file = r'C:\Users\freedom\Desktop\基金问答智能体\data\基金问题.jsonl'
with open(fund_file, 'r', encoding='utf-8') as f:
    fund_ids = set(json.loads(l)['id'] for l in f if l.strip())

user_ids = [21, 40, 46, 47, 59, 85, 175, 249, 502, 660, 800, 543]
print("那12道题在基金文件中的情况：")
for qid in user_ids:
    print(f"  id={qid}: {'IN_FUND' if qid in fund_ids else 'NOT_IN_FUND'}")

print(f"\n基金文件总题数: {len(fund_ids)}")
