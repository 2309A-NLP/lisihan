#!/usr/bin/env python3
"""
工单6 验收标准 — 工具选择准确率基准测试
需求：工具选择准确率 ≥ 90%，各工具功能准确率 ≥ 95%

用法：
  python benchmark_accuracy.py [--iterations 10] [--method agent|keyword] [--verbose]

测试用例覆盖5个工具的所有场景，统计准确率。
"""

import sys
import os
import json
import time
import argparse
from typing import List, Tuple, Set, Dict
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import Agent
from core.json_utils import extract_decision

# ============================================================
# 1. 测试用例集
# ============================================================

TEST_CASES: List[Tuple[str, List[str]]] = [
    # --- 记账本 (ledger_query) ---
    ("记录今天的支出100元", ["ledger_query"]),
    ("查一下我上个月花了多少钱", ["ledger_query"]),
    ("统计一下这个月的收入", ["ledger_query"]),
    ("帮我把昨天买菜的200元记上", ["ledger_query"]),
    ("看看我的消费流水", ["ledger_query"]),
    ("这个月餐饮花了多少", ["ledger_query"]),
    ("帮我查一下上周的支出", ["ledger_query"]),

    # --- 日程提醒 (schedule_query) ---
    ("提醒我明天下午3点开会", ["schedule_query"]),
    ("查一下我明天的日程安排", ["schedule_query"]),
    ("帮我设置一个周五的提醒", ["schedule_query"]),
    ("看看这周有哪些会议", ["schedule_query"]),
    ("取消周二的日程", ["schedule_query"]),
    ("下周一有什么安排", ["schedule_query"]),

    # --- 文生图 (image_generate) ---
    ("帮我画一只可爱的猫咪", ["image_generate"]),
    ("生成一张夏天的海滩图片", ["image_generate"]),
    ("绘制一幅山水画", ["image_generate"]),
    ("帮我做一张星空图", ["image_generate"]),
    ("生成一个卡通头像", ["image_generate"]),
    ("画一幅油画风格的风景", ["image_generate"]),

    # --- 基金数据 (fund_query) ---
    ("查询000001基金的最新净值", ["fund_query"]),
    ("帮我对比一下这两只基金", ["fund_query"]),
    ("看看最近收益最高的基金", ["fund_query"]),
    ("查一下白酒板块的基金表现", ["fund_query"]),
    ("基金A和基金B哪个好", ["fund_query"]),
    ("今天的基金行情怎么样", ["fund_query"]),

    # --- 招股书 (prospectus_query) ---
    ("查询某公司招股说明书中的财务数据", ["prospectus_query"]),
    ("招股说明书里提到的募投项目有哪些", ["prospectus_query"]),
    ("查一下这家公司的营收情况", ["prospectus_query"]),
    ("招股书中的风险因素是什么", ["prospectus_query"]),
    ("对比两家公司的招股说明书", ["prospectus_query"]),
    ("公司的股权结构是什么样的", ["prospectus_query"]),

    # --- 无需工具 (问候/闲聊) ---
    ("你好", []),
    ("谢谢", []),
    ("你是谁", []),
    ("再见", []),
    ("有什么功能", []),
    ("今天天气怎么样", []),  # 不支持的功能

    # --- 多工具协同 ---
    ("查我上个月花了多少钱，再看看000001基金的净值", ["ledger_query", "fund_query"]),
    ("帮我看看明天的日程和今天花了多少钱", ["schedule_query", "ledger_query"]),
    ("生成一张图，并查一下000001基金的净值", ["image_generate", "fund_query"]),
    ("记录今天午餐花了50，提醒我明天开会", ["ledger_query", "schedule_query"]),

    # --- 边界/歧义测试 ---
    ("帮我查一下", []),  # 意图不明确
    ("记录一下", []),  # 意图不明确
    ("画", []),  # 意图不明确
]

# 扩展测试用例：包含参数提取验证
PARAM_TEST_CASES: List[Tuple[str, List[str], Dict]] = [
    ("记录今天支出100元", ["ledger_query"], {"amount": "100", "date": "today"}),
    ("提醒我明天下午3点开会", ["schedule_query"], {"time": "15:00", "date": "tomorrow"}),
    ("查询000001基金净值", ["fund_query"], {"code": "000001"}),
]

# ============================================================
# 2. 关键词映射（回退方案）
# ============================================================

KEYWORD_MAP = {
    "ledger_query": [
        "记账", "账本", "收支", "花费", "支出", "收入", "花了",
        "流水", "消费", "记", "账单", "金额", "餐饮", "上周"
    ],
    "schedule_query": [
        "日程", "提醒", "会议", "安排", "日历", "明天", "后天",
        "下周", "周一", "周二", "周三", "周四", "周五"
    ],
    "image_generate": [
        "生成", "图片", "图", "画", "绘制", "做一张", "一张",
        "画一幅", "插图", "设计", "创作"
    ],
    "fund_query": [
        "基金", "净值", "收益率", "对比基金", "板块", "行情",
        "股票", "投资", "涨幅", "跌幅"
    ],
    "prospectus_query": [
        "招股", "说明书", "prospectus", "募投", "公司", "营收",
        "财务数据", "股权", "股东", "上市", "发行"
    ],
}


def keyword_match(text: str) -> Set[str]:
    """
    关键词匹配回退

    Args:
        text: 用户输入文本

    Returns:
        匹配的工具名集合
    """
    text = text.lower()
    found = set()
    for tool, keywords in KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            found.add(tool)
    return found


# ============================================================
# 3. Agent 模拟测试
# ============================================================

class AgentTester:
    """
    模拟 Agent 的意图识别流程

    包含：
    1. LLM 调用（通过 Agent 类）
    2. 关键词回退
    3. 结果合并
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.agent = None
        if use_llm:
            try:
                # 加载配置
                import yaml
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config", "config.yaml"
                )
                llm_config = {}
                if os.path.exists(config_path):
                    with open(config_path, encoding="utf-8") as f:
                        full = yaml.safe_load(f) or {}
                        llm_config = full.get("llm", {})

                # 创建 Agent（但不需要连接 MCP）
                mcp_script = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "mcp_server.py"
                )
                self.agent = Agent(mcp_script, llm_config)
                print(f"✓ Agent 初始化成功 (LLM: {llm_config.get('model', 'N/A')})")
            except Exception as e:
                print(f"⚠️ Agent 初始化失败，将使用关键词回退: {e}")
                self.use_llm = False

    def predict(self, query: str) -> Set[str]:
        """
        预测应该使用的工具

        Args:
            query: 用户输入

        Returns:
            预测的工具名集合
        """
        # 策略1: 使用 Agent 的 LLM 意图识别
        if self.use_llm and self.agent:
            try:
                # 构建 Prompt
                prompt = self.agent.PROMPT_TEMPLATE.format(
                    tool_descriptions=self.agent._describe_tools(),
                    conversation_history=self.agent._format_history(),
                    user_input=query,
                )

                # 调用 LLM
                llm_response = self.agent._call_llm(prompt, timeout_sec=5)

                # 解析结果
                decision = extract_decision(llm_response)
                tools = [t["name"] for t in decision.get("tools", [])]

                if tools:
                    return set(tools)
            except Exception as e:
                print(f"⚠️ LLM 预测失败: {e}")

        # 策略2: 关键词回退
        return keyword_match(query)


# ============================================================
# 4. 测试执行
# ============================================================

def run_accuracy_test(
        method: str = "agent",
        iterations: int = 1,
        verbose: bool = False
) -> Tuple[float, List[Tuple[str, List[str], List[str], bool]]]:
    """
    运行准确率测试

    Args:
        method: "agent" 或 "keyword"
        iterations: 每个用例重复次数
        verbose: 是否显示详细信息

    Returns:
        (准确率, 详细结果列表)
    """
    # 初始化测试器
    use_llm = (method == "agent")
    tester = AgentTester(use_llm=use_llm)

    total = 0
    correct = 0
    details = []

    # 准备测试用例（支持多次迭代）
    test_cases = []
    for _ in range(iterations):
        test_cases.extend(TEST_CASES)

    print(f"\n运行测试: {len(test_cases)} 个用例")
    print("-" * 60)

    for idx, (query, expected) in enumerate(test_cases, 1):
        expected_set = set(expected)

        # 预测
        if method == "keyword":
            result = keyword_match(query)
        else:
            result = tester.predict(query)

        # 判断是否完全匹配
        is_correct = (result == expected_set)
        if is_correct:
            correct += 1
        total += 1

        details.append((query, expected, list(result), is_correct))

        # 显示进度
        if verbose or idx % 10 == 0:
            status = "✅" if is_correct else "❌"
            print(f"  {status} [{idx:3d}/{len(test_cases)}] {query[:30]:30s} → {list(result)}")

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, details


def calculate_tool_stats(details: List[Tuple]) -> Dict:
    """
    计算每个工具的准确率统计

    Args:
        details: 测试详情列表

    Returns:
        工具统计字典
    """
    stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for query, expected, result, ok in details:
        for tool in expected:
            stats[tool]["total"] += 1
            if ok and tool in result:
                stats[tool]["correct"] += 1

    return dict(stats)


# ============================================================
# 5. 主入口
# ============================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="工具选择准确率基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python benchmark_accuracy.py                    # 使用 Agent 方法测试
  python benchmark_accuracy.py --iterations 3    # 每个用例测试3次
  python benchmark_accuracy.py --method keyword  # 仅使用关键词匹配
  python benchmark_accuracy.py --verbose         # 显示详细输出
        """
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="每个用例重复次数 (默认: 1)"
    )
    parser.add_argument(
        "--method",
        choices=["agent", "keyword"],
        default="agent",
        help="测试方法: agent=LLM+关键词, keyword=仅关键词 (默认: agent)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="保存结果到 JSON 文件"
    )
    args = parser.parse_args()

    # ===== 打印标题 =====
    print("=" * 70)
    print("  工单6 — 工具选择准确率基准测试")
    print("  需求: 工具选择准确率 ≥ 90%，各工具功能准确率 ≥ 95%")
    print("=" * 70)
    print(f"  方法: {args.method}")
    print(f"  迭代次数: {args.iterations}")
    print(f"  用例总数: {len(TEST_CASES)} x {args.iterations} = {len(TEST_CASES) * args.iterations}")
    print("=" * 70)

    # ===== 运行测试 =====
    t_start = time.time()
    accuracy, details = run_accuracy_test(
        method=args.method,
        iterations=args.iterations,
        verbose=args.verbose
    )
    t_elapsed = time.time() - t_start

    total = len(details)
    correct = int(accuracy / 100 * total + 0.5)  # 四舍五入

    # ===== 计算各工具统计 =====
    tool_stats = calculate_tool_stats(details)

    # ===== 输出结果 =====
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)

    # 总准确率
    print(f"\n总准确率: {accuracy:.1f}% ({correct}/{total})")
    status = "✅ 达标" if accuracy >= 90 else "❌ 不达标"
    print(f"验收标准: {status} (标准 ≥ 90%)")
    print(f"耗时: {t_elapsed:.2f}s")

    # 各工具准确率
    print("\n--- 各工具准确率 ---")
    all_tools_pass = True
    for tool in sorted(tool_stats.keys()):
        stats = tool_stats[tool]
        rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        status = "✅" if rate >= 95 else "❌"
        print(f"  {status} {tool:25s}: {rate:5.1f}% ({stats['correct']:3d}/{stats['total']:3d})")
        if rate < 95:
            all_tools_pass = False

    # ===== 显示失败的用例 =====
    failures = [(q, e, r) for q, e, r, ok in details if not ok]

    if failures:
        print(f"\n--- 失败的用例 ({len(failures)}个) ---")
        # 按工具分组显示
        failure_by_tool = defaultdict(list)
        for q, e, r in failures:
            for tool in e:
                failure_by_tool[tool].append((q, e, r))
            if not e:  # 期望无工具但识别出工具
                failure_by_tool["no_tool"].append((q, e, r))

        for tool, cases in failure_by_tool.items():
            print(f"\n  [{tool}] ({len(cases)}个失败)")
            for q, e, r in cases[:5]:  # 每个工具最多显示5个
                print(f"    ❌ 输入: {q[:50]}")
                print(f"       期望: {e}")
                print(f"       实际: {r}")
            if len(cases) > 5:
                print(f"    ... 还有 {len(cases) - 5} 个失败")

    # ===== 验收结论 =====
    print("\n" + "=" * 70)
    print("📋 验收结论")
    print("=" * 70)

    total_pass = accuracy >= 90
    tools_pass = all_tools_pass

    if total_pass and tools_pass:
        print("✅ 所有验收标准已达标!")
        print("   - 工具选择准确率 ≥ 90%: {} ({:.1f}%)".format(
            "✅" if accuracy >= 90 else "❌", accuracy
        ))
        for tool, stats in sorted(tool_stats.items()):
            rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"   - {tool} ≥ 95%: {'✅' if rate >= 95 else '❌'} ({rate:.1f}%)")
    else:
        print("❌ 部分验收标准未达标:")
        if not total_pass:
            print(f"   - 工具选择准确率: {accuracy:.1f}% (标准 ≥ 90%)")
        if not tools_pass:
            print(f"   - 部分工具准确率 < 95%，请检查上表")

    # ===== 保存结果 =====
    if args.save_results:
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": args.method,
            "iterations": args.iterations,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "tool_stats": tool_stats,
            "failures": [
                {"query": q, "expected": e, "actual": r}
                for q, e, r, ok in details if not ok
            ]
        }
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.save_results}")

    print("=" * 70)


# ============================================================
# 6. 模块导入支持
# ============================================================

if __name__ == "__main__":
    main()