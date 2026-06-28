"""找出被移除的9道题"""
import json

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    all_q = {d['id']: d['question'] for d in [json.loads(l) for l in f if l.strip()]}

with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    v3_ids = set(json.loads(l)['id'] for l in f if l.strip())

# 模拟v2规则（旧版classify_questions.py）的招股书ID
# 旧规则：公司名+无基金关键词 → prospectus
import re
v2_ids = set()
for qid, question in all_q.items():
    q = question
    # 旧版的PROSPECTUS_STRONG
    old_kw = ["招股","招股说明书","招股意向书","本次发行","发行人","发起人","股权转让",
              "控股股东","占公司总股份","占公司总股本","变更设立","变更设立时",
              "总资产周转率","竞争优势","产品研发","近三年","存货","占流动资产",
              "募集资金","募投项目","前十大股东","毛利率","净利率","产能","产量","销量",
              "核心技术","研发投入","主营业务收入","营业收入构成","前五大客户","负责产品",
              "主营业务","营业范围","净资产","总资产",
              "公司的","该公司","有限公司","股份有限公司","有限责任公司"]
    is_p = any(k in q for k in old_kw)
    if is_p:
        v2_ids.add(qid)

removed = v2_ids - v3_ids
print(f"被移除的 {len(removed)} 道题：\n")
for rid in sorted(removed):
    print(f"  id={rid}: {all_q[rid][:120]}")
    print()
