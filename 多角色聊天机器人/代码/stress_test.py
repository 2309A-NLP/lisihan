"""
压力测试脚本 - 完整修正版（无图形依赖）
正确计算QPS、吞吐量、响应时间等性能指标
"""

import requests
import time
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime

API = "http://127.0.0.1:8000"

# 测试配置
CONFIG = {
    "concurrent_users": [1, 5, 10, 20, 50],  # 并发用户数
    "requests_per_user": 10,  # 每个用户请求次数
    "questions": [
        "高血压要注意什么？",
        "民法典第1043条是什么？",
        "什么是因材施教？",
        "现在进行时怎么用？",
        "如何预防感冒？",
        "什么是人工智能？",
    ],
}


# Function: Run load tests and produce performance summaries.
class LoadTester:
    """负载测试器"""

    # Function: Initialize instance fields and runtime dependencies.
    def __init__(self, api_url=API):
        self.api_url = api_url
        self.token = None
        self.all_results = []

    # Function: Validate credentials, create a token, and update login metadata.
    def login(self):
        """登录获取token"""
        try:
            resp = requests.post(f"{self.api_url}/api/user/login",
                               json={"username": "admin", "password": "admin123"},
                               timeout=5)
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                print("✅ 登录成功")
                return self.token
            else:
                print(f"❌ 登录失败: {resp.status_code}")
        except Exception as e:
            print(f"❌ 登录异常: {e}")
        return None

    # Function: Send one chat request during load testing.
    def chat_request(self, question: str, role_id: int = 1, request_id: int = 0) -> dict:
        """发送聊天请求并记录详细指标"""
        start_time = time.time()
        timestamp = start_time

        try:
            resp = requests.post(f"{self.api_url}/api/chat",
                               json={"user_id": 1, "role_id": role_id, "message": question},
                               headers={"Authorization": f"Bearer {self.token}"},
                               timeout=60)

            elapsed_ms = (time.time() - start_time) * 1000

            # 尝试解析响应内容
            response_text = ""
            try:
                response_data = resp.json()
                response_text = response_data.get("response", "")[:100]
            except:
                response_text = resp.text[:100]

            return {
                "success": resp.status_code == 200,
                "latency_ms": elapsed_ms,
                "status_code": resp.status_code,
                "timestamp": timestamp,
                "question": question[:30],
                "request_id": request_id,
                "response_preview": response_text,
                "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}"
            }
        except requests.exceptions.Timeout:
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency_ms": elapsed_ms,
                "status_code": 408,
                "timestamp": timestamp,
                "question": question[:30],
                "request_id": request_id,
                "response_preview": "",
                "error": "Timeout"
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency_ms": elapsed_ms,
                "status_code": 0,
                "timestamp": timestamp,
                "question": question[:30],
                "request_id": request_id,
                "response_preview": "",
                "error": str(e)
            }

    # Function: Run a fixed concurrency load test.
    def run_load_test(self, concurrent: int, requests_per_user: int) -> dict:
        """运行负载测试 - 正确计算QPS"""
        print(f"\n{'='*60}")
        print(f"📊 测试并发数: {concurrent}")
        print(f"{'='*60}")

        if not self.token and not self.login():
            return {"error": "登录失败"}

        results = []
        questions = CONFIG["questions"]
        total_requests = concurrent * requests_per_user

        print(f"   总请求数: {total_requests}")
        print(f"   每用户请求: {requests_per_user}")
        print(f"   开始测试...")

        # 记录测试开始时间
        test_start_time = time.time()
        start_timestamp = test_start_time

        # 执行并发请求
        request_id = 0
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = []
            for user_id in range(concurrent):
                for req_idx in range(requests_per_user):
                    question = questions[(request_id) % len(questions)]
                    role_id = (request_id % 5) + 1
                    futures.append(executor.submit(self.chat_request, question, role_id, request_id))
                    request_id += 1

            # 收集结果
            for future in as_completed(futures):
                results.append(future.result())

        # 记录测试结束时间
        test_end_time = time.time()
        total_duration = test_end_time - test_start_time

        # 统计分析
        success_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        latencies = [r["latency_ms"] for r in success_results]

        if not latencies:
            return {
                "error": "所有请求失败",
                "total": len(results),
                "success": 0,
                "failed": len(results)
            }

        # 计算百分位数
        sorted_latencies = sorted(latencies)

        # Function: Calculate a percentile value from sorted latency data.
        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data)-1)]

        # 计算时间分布（用于QPS曲线）
        time_windows = defaultdict(int)
        for r in results:
            window = int(r["timestamp"] - start_timestamp)
            time_windows[window] += 1

        # 计算QPS
        qps_total = len(success_results) / total_duration

        # 计算峰值QPS（1秒窗口内的最大请求数）
        peak_qps = max(time_windows.values()) if time_windows else 0

        # 计算吞吐量（KB/s）
        total_response_size = sum(len(r.get("response_preview", "")) for r in success_results)
        throughput = total_response_size / total_duration / 1024  # KB/s

        return {
            "concurrent": concurrent,
            "total_requests": len(results),
            "success_count": len(success_results),
            "failed_count": len(failed_results),
            "success_rate": len(success_results) / len(results) * 100,
            "total_duration": total_duration,
            "qps_avg": qps_total,
            "qps_peak": peak_qps,
            "throughput_kb_s": throughput,
            "latency_avg_ms": statistics.mean(latencies),
            "latency_median_ms": statistics.median(latencies),
            "latency_min_ms": min(latencies),
            "latency_max_ms": max(latencies),
            "latency_p50_ms": percentile(sorted_latencies, 50),
            "latency_p90_ms": percentile(sorted_latencies, 90),
            "latency_p95_ms": percentile(sorted_latencies, 95),
            "latency_p99_ms": percentile(sorted_latencies, 99),
            "errors": [r["error"] for r in failed_results if r.get("error")][:10],  # 最多10个错误
            "raw_results": results  # 保存原始结果用于后续分析
        }

    # Function: Run a duration-based load test at a target QPS.
    def run_continuous_test(self, duration_seconds: int = 30, target_qps: int = 10):
        """持续压力测试 - 恒定QPS"""
        print(f"\n{'='*60}")
        print(f"🔥 持续压力测试 ({duration_seconds}秒, 目标QPS={target_qps})")
        print(f"{'='*60}")

        if not self.token and not self.login():
            print("❌ 登录失败")
            return None

        questions = CONFIG["questions"]
        results = []
        start_time = time.time()
        request_count = 0
        interval = 1.0 / target_qps  # 请求间隔

        next_request_time = start_time

        print(f"   开始发送请求...")

        while time.time() - start_time < duration_seconds:
            # 控制请求速率
            current_time = time.time()
            if current_time < next_request_time:
                time.sleep(max(0, next_request_time - current_time))

            # 发送请求
            question = questions[request_count % len(questions)]
            role_id = (request_count % 5) + 1
            result = self.chat_request(question, role_id, request_count)
            results.append(result)
            request_count += 1

            next_request_time = start_time + (request_count * interval)

            # 进度显示
            if request_count % 20 == 0:
                elapsed = time.time() - start_time
                current_qps = request_count / elapsed if elapsed > 0 else 0
                print(f"   进度: {request_count} 请求, 当前QPS: {current_qps:.1f}")

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r["success"])
        latencies = [r["latency_ms"] for r in results if r["success"]]

        print(f"\n📊 持续测试结果:")
        print(f"   总请求数: {request_count}")
        print(f"   成功数: {success_count}")
        print(f"   成功率: {success_count/request_count*100:.1f}%")
        print(f"   实际QPS: {request_count/elapsed:.1f} (目标: {target_qps})")
        if latencies:
            print(f"   平均延迟: {statistics.mean(latencies):.0f}ms")
            print(f"   最小延迟: {min(latencies):.0f}ms")
            print(f"   最大延迟: {max(latencies):.0f}ms")
            print(f"   P95延迟: {statistics.quantiles(latencies, n=20)[18]:.0f}ms")

        return {
            "duration": elapsed,
            "total_requests": request_count,
            "success_count": success_count,
            "qps_actual": request_count / elapsed,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "results": results
        }

    # Function: Generate a text performance report.
    def generate_report(self, test_results):
        """生成详细测试报告"""
        print("\n" + "="*80)
        print("📊 压力测试详细报告")
        print("="*80)

        # 汇总表格
        print(f"\n{'并发数':<8} {'总请求':<8} {'成功':<8} {'成功率':<10} {'平均QPS':<10} {'峰值QPS':<10} {'平均延迟(ms)':<12} {'P95(ms)':<10}")
        print("-"*80)

        for r in test_results:
            if "error" not in r:
                print(f"{r['concurrent']:<8} "
                      f"{r['total_requests']:<8} "
                      f"{r['success_count']:<8} "
                      f"{r['success_rate']:<9.1f}% "
                      f"{r['qps_avg']:<10.2f} "
                      f"{r['qps_peak']:<10} "
                      f"{r['latency_avg_ms']:<12.0f} "
                      f"{r['latency_p95_ms']:<10.0f}")

        # 性能评估
        print("\n" + "="*80)
        print("🏆 性能评估")
        print("="*80)

        if test_results:
            # 找到最佳QPS（排除错误的结果）
            valid_results = [r for r in test_results if "error" not in r]
            if valid_results:
                max_qps = max(r["qps_avg"] for r in valid_results)
                min_latency = min(r["latency_avg_ms"] for r in valid_results)
                best_success_rate = max(r["success_rate"] for r in valid_results)

                print(f"\n  最大吞吐量(QPS): {max_qps:.2f} 请求/秒")
                print(f"  最佳平均延迟: {min_latency:.0f}ms")
                print(f"  最高成功率: {best_success_rate:.1f}%")

                # QPS评级
                print(f"\n  📈 QPS评级:")
                if max_qps >= 50:
                    print("  ⭐⭐⭐⭐⭐ QPS优秀 (≥50)")
                elif max_qps >= 30:
                    print("  ⭐⭐⭐⭐ QPS良好 (≥30)")
                elif max_qps >= 10:
                    print("  ⭐⭐⭐ QPS中等 (≥10)")
                else:
                    print("  ⭐⭐ QPS待提升 (<10)")

                # 延迟评级
                print(f"\n  ⏱️  延迟评级:")
                if min_latency <= 2000:
                    print("  ⭐⭐⭐⭐⭐ 延迟优秀 (≤2秒)")
                elif min_latency <= 5000:
                    print("  ⭐⭐⭐⭐ 延迟良好 (≤5秒)")
                elif min_latency <= 10000:
                    print("  ⭐⭐⭐ 延迟中等 (≤10秒)")
                else:
                    print("  ⭐⭐ 延迟待优化 (>10秒)")

                # 稳定性评级
                print(f"\n  🎯 稳定性评级:")
                if best_success_rate >= 99:
                    print("  ⭐⭐⭐⭐⭐ 非常稳定 (≥99%)")
                elif best_success_rate >= 95:
                    print("  ⭐⭐⭐⭐ 稳定 (≥95%)")
                elif best_success_rate >= 90:
                    print("  ⭐⭐⭐ 基本稳定 (≥90%)")
                else:
                    print("  ⭐⭐ 稳定性待提升 (<90%)")

        # 瓶颈分析
        print("\n" + "="*80)
        print("🔍 瓶颈分析")
        print("="*80)

        for r in test_results:
            if "error" not in r and r["qps_avg"] > 0:
                # 计算效率（实际QPS / 理论最大QPS）
                theoretical_max = 1000 / r["latency_avg_ms"] * r["concurrent"]
                efficiency = r["qps_avg"] / theoretical_max * 100 if theoretical_max > 0 else 0

                print(f"\n  并发数 {r['concurrent']}:")
                print(f"    理论最大QPS: {theoretical_max:.1f}")
                print(f"    实际QPS: {r['qps_avg']:.1f}")
                print(f"    并发效率: {efficiency:.1f}%")

                if efficiency < 50:
                    print(f"    ⚠️  效率较低，可能存在锁竞争或资源瓶颈")
                    print(f"    建议: 检查数据库连接池、Redis连接数、API限流设置")
                elif efficiency < 80:
                    print(f"    ✓ 效率中等")
                    print(f"    建议: 可考虑增加缓存或优化数据库查询")
                else:
                    print(f"    ✅ 效率优秀")
                    print(f"    建议: 系统扩展性良好，可继续增加并发")

        # 响应时间分布
        print("\n" + "="*80)
        print("📈 响应时间分布")
        print("="*80)

        for r in test_results:
            if "error" not in r:
                print(f"\n  并发数 {r['concurrent']}:")
                print(f"    最小: {r['latency_min_ms']:.0f}ms")
                print(f"    P50:  {r['latency_p50_ms']:.0f}ms")
                print(f"    P90:  {r['latency_p90_ms']:.0f}ms")
                print(f"    P95:  {r['latency_p95_ms']:.0f}ms")
                print(f"    P99:  {r['latency_p99_ms']:.0f}ms")
                print(f"    最大: {r['latency_max_ms']:.0f}ms")

                # 判断是否有长尾延迟
                if r['latency_p99_ms'] > r['latency_p95_ms'] * 2:
                    print(f"    ⚠️  存在长尾延迟，P99是P95的{r['latency_p99_ms']/r['latency_p95_ms']:.1f}倍")

    # Function: Save load test results as JSON.
    def save_results_to_file(self, results, filename="test_results.json"):
        """保存测试结果到文件"""
        # 转换数据为可序列化格式
        serializable_results = []
        for r in results:
            if "raw_results" in r:
                # 简化原始结果，只保留关键信息
                r_copy = r.copy()
                r_copy["raw_results_summary"] = {
                    "total": len(r["raw_results"]),
                    "success": sum(1 for rr in r["raw_results"] if rr["success"]),
                    "avg_latency": statistics.mean([rr["latency_ms"] for rr in r["raw_results"] if rr["success"]]) if any(rr["success"] for rr in r["raw_results"]) else 0
                }
                del r_copy["raw_results"]
                serializable_results.append(r_copy)
            else:
                serializable_results.append(r)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 测试结果已保存到: {filename}")

    # Function: Print an ASCII chart of load test metrics.
    def print_ascii_chart(self, results):
        """打印ASCII图表"""
        print("\n" + "="*80)
        print("📊 性能可视化 (ASCII)")
        print("="*80)

        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return

        # QPS对比图
        print("\n📈 QPS vs 并发数:")
        max_qps = max(r["qps_avg"] for r in valid_results)
        for r in valid_results:
            bar_length = int(r["qps_avg"] / max_qps * 40)
            bar = "█" * bar_length
            print(f"  {r['concurrent']:3} 并发: {bar} {r['qps_avg']:.1f} QPS")

        # 延迟对比图
        print("\n⏱️  延迟 vs 并发数:")
        max_latency = max(r["latency_avg_ms"] for r in valid_results)
        for r in valid_results:
            bar_length = int(r["latency_avg_ms"] / max_latency * 40)
            bar = "█" * bar_length
            print(f"  {r['concurrent']:3} 并发: {bar} {r['latency_avg_ms']:.0f}ms")

        # 效率对比图
        print("\n🎯 并发效率:")
        for r in valid_results:
            theoretical_max = 1000 / r["latency_avg_ms"] * r["concurrent"]
            efficiency = r["qps_avg"] / theoretical_max * 100 if theoretical_max > 0 else 0
            bar_length = int(efficiency / 100 * 40)
            bar = "█" * bar_length
            print(f"  {r['concurrent']:3} 并发: {bar} {efficiency:.1f}%")


# Function: Run the script entry point.
def main():
    """主函数"""
    print("="*80)
    print("🚀 RAG系统压力测试工具 v2.0 (无图形依赖版)")
    print("="*80)

    # 1. 健康检查
    try:
        resp = requests.get(f"{API}/health", timeout=5)
        if resp.status_code != 200:
            print("❌ 服务未启动，请先运行 python main.py")
            return
        print("✅ 服务健康检查通过")
    except Exception as e:
        print(f"❌ 服务未启动或无法连接: {e}")
        return

    # 2. 初始化测试器
    tester = LoadTester(API)

    # 3. 快速基准测试
    print("\n📌 快速基准测试")
    if tester.login():
        # 预热
        print("   预热中...")
        for i in range(3):
            tester.chat_request("测试问题", 1, i)

        # 正式测试
        start = time.time()
        result = tester.chat_request("高血压要注意什么？", 1, 0)
        elapsed = (time.time() - start) * 1000
        if result["success"]:
            print(f"   ✅ 单次请求耗时: {elapsed:.0f}ms")
            print(f"   📝 响应预览: {result['response_preview'][:80]}...")
        else:
            print(f"   ❌ 请求失败: {result['error']}")

    # 4. 询问是否运行完整测试
    print("\n" + "="*80)
    run_full = input("是否运行完整压力测试？(y/n, 默认n): ").lower()

    all_results = []
    if run_full == 'y':
        # 运行不同并发级别的测试
        for concurrent in CONFIG["concurrent_users"]:
            result = tester.run_load_test(concurrent, CONFIG["requests_per_user"])
            if "error" not in result:
                all_results.append(result)
                print(f"\n   ✅ 并发={concurrent}:")
                print(f"      QPS={result['qps_avg']:.2f} (峰值={result['qps_peak']})")
                print(f"      平均延迟={result['latency_avg_ms']:.0f}ms")
                print(f"      成功率={result['success_rate']:.1f}%")
                print(f"      总耗时={result['total_duration']:.2f}秒")
            else:
                print(f"\n   ❌ 并发={concurrent}: {result['error']}")

        # 5. 生成报告
        if all_results:
            tester.generate_report(all_results)
            tester.print_ascii_chart(all_results)
            tester.save_results_to_file(all_results)
    else:
        # 只运行一个简单的测试
        print("\n运行快速测试...")
        result = tester.run_load_test(5, 5)
        if "error" not in result:
            all_results.append(result)
            tester.generate_report(all_results)
            tester.print_ascii_chart(all_results)

    # 6. 持续压力测试（可选）
    print("\n" + "="*80)
    run_continuous = input("是否运行持续压力测试？(y/n, 默认n): ").lower()
    if run_continuous == 'y':
        try:
            target_qps = int(input("目标QPS (默认10): ") or "10")
            duration = int(input("测试时长秒数 (默认30): ") or "30")
            tester.run_continuous_test(duration, target_qps)
        except ValueError:
            print("输入无效，使用默认值")
            tester.run_continuous_test(30, 10)

    print("\n" + "="*80)
    print("✅ 压力测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()
