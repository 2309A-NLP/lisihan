"""
从 fund_results.jsonl 提取纯净提交格式
输出: output/fund_submit.jsonl (每行 {"id": N, "answer": "..."})
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(BASE, "..", "output", "batch_results.jsonl")
OUT_PATH = os.path.join(BASE, "..", "output", "submission.jsonl")

def extract():
    if not os.path.exists(IN_PATH):
        print(f"未找到 {IN_PATH}，请先运行 batch_test.py")
        return
    
    count = 0
    with open(IN_PATH, 'r', encoding='utf-8') as f_in, \
         open(OUT_PATH, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.strip():
                r = json.loads(line)
                submit = {"id": r["id"], "answer": r["answer"]}
                f_out.write(json.dumps(submit, ensure_ascii=False) + '\n')
                count += 1
    
    print(f"已生成提交文件: {OUT_PATH}")
    print(f"共 {count} 条记录")

if __name__ == '__main__':
    extract()
