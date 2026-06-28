"""Sample questions every 50 lines to understand the mix"""
import json

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

intervals = list(range(0, len(lines), 50)) + [len(lines)-1]
for idx in intervals:
    d = json.loads(lines[idx])
    q = d['question'][:120]
    print(f'--- id={d["id"]} ---')
    print(q)
    print()
