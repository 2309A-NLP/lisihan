"""完整核对分类结果"""
import json

# 读所有题
with open(r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json', 'r', encoding='utf-8') as f:
    all_q = [json.loads(l) for l in f if l.strip()]

# 读招股书分类
with open(r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl', 'r', encoding='utf-8') as f:
    pros_ids = set(json.loads(l)['id'] for l in f if l.strip())

# 读基金分类（反向）
fund_ids = [d['id'] for d in all_q if d['id'] not in pros_ids]

print(f"总题数: {len(all_q)}")
print(f"招股书: {len(pros_ids)}")
print(f"基金: {len(fund_ids)}")
print(f"合计: {len(pros_ids) + len(fund_ids)}")
print()

# 用户说应该有591道基金 → 1000-591=409道招股书
# 目前只有397，差12道
# 检查那12道是不是漏了
user_prospectus_ids = [21, 40, 46, 47, 59, 85, 175, 249, 502, 660, 800, 543]

print("=== 用户指出应归招股书的那12道 ===")
for qid in user_prospectus_ids:
    status = "IN_PROSPECTUS" if qid in pros_ids else "IN_FUND"
    q = next((d['question'][:100] for d in all_q if d['id'] == qid), 'NOT FOUND')
    print(f"id={qid} {status}: {q}")

print()

# 找找还有没有其他疑似被归为基金的招股书题
print("=== 基金分类中疑似招股书（含公司名但无基金词）===")
count = 0
for d in all_q:
    if d['id'] in pros_ids:
        continue
    q = d['question']
    # 有公司名但无基金词
    has_company = ('股份有限公司' in q or '有限公司' in q)
    has_fund_words = any(w in q for w in ['基金', '代码', '涨跌幅', '收盘价', '成交', '港股', '中证'])
    if has_company and not has_fund_words:
        count += 1
        print(f"id={d['id']}: {q[:120]}")
        if count >= 20:
            print(f"... 还有更多，共?个")
            break
if count == 0:
    print("  (无)")
