"""
招股书问答智能体 - 批量测试脚本
对招股书问题批量运行 RAG 回答，每20题保存一次
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 修正 Windows 终端编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from react_agent import react_agent, init

BASE = os.path.dirname(os.path.abspath(__file__))
Q_PATH = os.path.join(BASE, "..", "data", "招股书问题.jsonl")
OUT_DIR = os.path.join(BASE, "..", "output")
OUT_PATH = os.path.join(OUT_DIR, "prospectus_results.jsonl")
SAVE_EVERY = 20

os.makedirs(OUT_DIR, exist_ok=True)

def load_questions():
    questions = []
    with open(Q_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def main():
    print("=" * 50)
    print("招股书问答智能体 - 批量测试")
    print("=" * 50)

    # 初始化索引
    print("初始化 RAG 索引...")
    init()
    print("索引加载完成")

    # 加载问题
    questions = load_questions()
    print(f"共 {len(questions)} 道招股书问题")

    # 已有进度
    results = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results[r['id']] = r
        print(f"已有进度: {len(results)}/{len(questions)}")

    remaining = [q for q in questions if q['id'] not in results]
    if not remaining:
        print("全部完成！")
        return

    print(f"待处理: {len(remaining)} 题")

    for i, q in enumerate(remaining):
        qid = q['id']
        question = q['question']
        print(f"[{i+1}/{len(remaining)}] #{qid}: {question[:50]}...", end=" ", flush=True)

        try:
            answer, _ = react_agent(question)
            results[qid] = {"id": qid, "question": question, "answer": answer}
            print("OK" if not answer.startswith("[") else "FAIL")
        except Exception as e:
            results[qid] = {"id": qid, "question": question, "answer": f"[异常:{e}]"}
            print(f" FAIL {e}")

        # 定期保存
        processed = len(results)
        if processed % SAVE_EVERY == 0:
            with open(OUT_PATH, 'w', encoding='utf-8') as f:
                for r in sorted(results.values(), key=lambda x: x['id']):
                    f.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + '\n')
            print(f"  [自动保存] {processed} 条")

        time.sleep(0.3)

    # 最终保存
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for r in sorted(results.values(), key=lambda x: x['id']):
            f.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + '\n')
    print(f"\n全部完成! 共 {len(results)} 条结果")
    print(f"输出: {OUT_PATH}")

if __name__ == "__main__":
    main()
