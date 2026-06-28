"""Check the 6 unclassified questions"""
import json

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

for line in lines:
    d = json.loads(line)
    if d['id'] in [501, 609, 776, 810, 861, 879]:
        q = d['question']
        print(f'id={d["id"]}: {q[:150]}')
        print()
