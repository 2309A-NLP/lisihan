import json
with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    ids = set(json.loads(l)['id'] for l in f if l.strip())
ids_to_check = [21, 40, 46, 47, 59, 85, 175, 249, 502, 660, 800, 543]
for qid in ids_to_check:
    print(f'id={qid}: {"YES" if qid in ids else "NO"}')
