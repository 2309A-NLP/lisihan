"""
RAG 系统精确评测（改进版）
"""

import requests
import time
from collections import defaultdict

API = "http://127.0.0.1:8000"

# 测试用例（使用更精确的评估标准）
test_cases = [
    # 医生角色
    {"question": "高血压要注意什么？", "role_id": 1,
     "keywords": ["低盐", "运动", "监测", "血压"],
     "min_recall": 0.5},  # 最低期望召回率

    {"question": "被狗咬了怎么办？", "role_id": 1,
     "keywords": ["冲洗", "疫苗", "就医", "伤口"],
     "min_recall": 0.5},

    {"question": "糖尿病怎么预防？", "role_id": 1,
     "keywords": ["饮食", "运动", "血糖", "控制"],
     "min_recall": 0.5},

    # 律师角色
    {"question": "民法典第1043条是什么？", "role_id": 2,
     "keywords": ["夫妻", "忠实", "尊重", "婚姻"],
     "min_recall": 0.5},



    {"question": "什么是因材施教？", "role_id": 4,
     "keywords": ["学生", "差异", "针对性", "个性"],
     "min_recall": 0.5},

    {"question": "什么是科学方法？", "role_id": 5,
     "keywords": ["观察", "实验", "验证"],
     "min_recall": 0.5},

    {"question": "现在进行时怎么用？", "role_id": 7,
     "keywords": ["正在", "be", "doing"],
     "min_recall": 0.5},
]

def evaluate_response(answer: str, keywords: list) -> dict:
    """改进的评估方法"""
    answer_lower = answer.lower()

    # 计算召回率（关键词命中率）
    hit_keywords = []
    for kw in keywords:
        if kw.lower() in answer_lower:
            hit_keywords.append(kw)

    recall = len(hit_keywords) / len(keywords) if keywords else 0

    # 改进的精确率：基于关键词的实际命中，不估算
    # 如果回答中包含所有关键词，精确率才高
    # 这里简化：精确率 = 命中关键词数 / (回答句子数/2)
    sentences = answer.split('。')
    precision = len(hit_keywords) / max(len(sentences) / 2, 1)
    precision = min(precision, 1.0)

    # F1分数
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "hit_keywords": hit_keywords,
        "missed_keywords": [kw for kw in keywords if kw not in hit_keywords]
    }

def run_evaluation():
    print("=" * 70)
    print("RAG 系统精确评测")
    print("=" * 70)

    # 登录
    login = requests.post(f"{API}/api/user/login",
        json={"username": "admin", "password": "admin123"})

    if login.status_code != 200:
        print("❌ 登录失败")
        return

    token = login.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ 登录成功\n")

    # 测试
    results = []
    total_time = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] 角色{tc['role_id']}: {tc['question']}")

        start = time.time()
        response = requests.post(f"{API}/api/chat",
            json={"user_id": 1, "role_id": tc["role_id"], "message": tc["question"]},
            headers=headers)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        if response.status_code == 200:
            answer = response.json().get("response", "")
            eval_res = evaluate_response(answer, tc["keywords"])

            results.append({
                "role_id": tc["role_id"],
                "question": tc["question"],
                "answer": answer[:200],
                "latency": elapsed,
                **eval_res,
                "min_recall": tc.get("min_recall", 0.5)
            })

            # 判断是否达标
            passed = eval_res["recall"] >= tc.get("min_recall", 0.5)
            status = "✅" if passed else "❌"

            print(f"  {status} 召回率: {eval_res['recall']:.0%} | "
                  f"精确率: {eval_res['precision']:.0%} | "
                  f"F1: {eval_res['f1']:.0%} | "
                  f"耗时: {elapsed:.0f}ms")

            if eval_res["missed_keywords"]:
                print(f"     未命中关键词: {eval_res['missed_keywords']}")
        else:
            print(f"  ❌ 请求失败")

    # 统计
    print("\n" + "=" * 70)
    print("评测总结")
    print("=" * 70)

    if not results:
        return

    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_latency = total_time / len(results)

    passed_count = sum(1 for r in results if r["recall"] >= r["min_recall"])

    print(f"✅ 通过率: {passed_count}/{len(results)} ({passed_count/len(results):.0%})")
    print(f"📊 平均召回率: {avg_recall:.1%}")
    print(f"📊 平均精确率: {avg_precision:.1%}")
    print(f"📊 平均F1分数: {avg_f1:.1%}")
    print(f"⏱️ 平均响应时间: {avg_latency:.0f}ms")

    # 改进建议
    print("\n💡 改进建议:")
    low_recall = [r for r in results if r["recall"] < 0.5]
    if low_recall:
        print(f"  - 召回率低的问题: {len(low_recall)}个")
        for r in low_recall:
            print(f"    · {r['question'][:30]}... (命中: {r['hit_keywords']})")
    else:
        print("  - 召回率良好！")

    if avg_latency > 10000:
        print("  - 响应时间偏慢(>10秒)，建议优化检索速度")

    print("\n📋 各角色表现:")
    role_stats = defaultdict(lambda: {"recall": [], "latency": []})
    for r in results:
        role_stats[r["role_id"]]["recall"].append(r["recall"])
        role_stats[r["role_id"]]["latency"].append(r["latency"])

    role_names = {1: "医生", 2: "律师", 4: "教师", 5: "科学家", 7: "英语助手"}
    for rid, stats in sorted(role_stats.items()):
        avg_rec = sum(stats["recall"]) / len(stats["recall"])
        avg_lat = sum(stats["latency"]) / len(stats["latency"])
        name = role_names.get(rid, f"角色{rid}")
        print(f"  {name}: 召回率={avg_rec:.0%}, 耗时={avg_lat:.0f}ms")

if __name__ == "__main__":
    run_evaluation()