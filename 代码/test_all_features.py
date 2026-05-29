#!/usr/bin/env python3
"""
测试所有功能的正常运行
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.rag import RAGSystem
from app.services.knowledge_service import KnowledgeService
from app.core.evaluation import RAGEvaluation

print("开始测试 RAG 系统功能...")

# 测试 1: 初始化 RAG 系统
print("\n1. 测试 RAG 系统初始化...")
try:
    rag_system = RAGSystem()
    print("✅ RAG 系统初始化成功")
    print(f"  - 知识库缓存大小: {len(rag_system.knowledge_cache)}")
    print(f"  - BM25 索引状态: {'已初始化' if rag_system.bm25_index else '未初始化'}")
except Exception as e:
    print(f"❌ RAG 系统初始化失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 2: 测试重排序功能
print("\n2. 测试重排序功能...")
try:
    test_query = "什么是高血压？"
    test_results = [
        {"content": "高血压是一种常见的慢性疾病，需要长期治疗。"},
        {"content": "高血压的症状包括头痛、头晕等。"},
        {"content": "高血压患者需要注意饮食，减少盐的摄入。"}
    ]
    reranked_results = rag_system.rerank_results(test_query, test_results)
    print("✅ 重排序功能测试成功")
    print(f"  - 重排序前: {[r['content'][:20] + '...' for r in test_results]}")
    print(f"  - 重排序后: {[r['content'][:20] + '...' for r in reranked_results]}")
except Exception as e:
    print(f"❌ 重排序功能测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 3: 测试混合检索功能
print("\n3. 测试混合检索功能...")
try:
    test_query = "高血压的治疗方法"
    search_results = rag_system.search_knowledge(test_query, top_k=3)
    print("✅ 混合检索功能测试成功")
    print(f"  - 检索结果数量: {len(search_results)}")
    for i, result in enumerate(search_results[:2]):
        print(f"  - 结果 {i+1}: {result['content'][:50] + '...'}")
except Exception as e:
    print(f"❌ 混合检索功能测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 测试知识库动态更新功能
print("\n4. 测试知识库动态更新功能...")
try:
    knowledge_service = KnowledgeService()
    # 构建知识库目录路径
    current_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(current_file)
    knowledge_dir = os.path.join(project_dir, 'app', 'knowledge', 'data')
    
    # 检查更新
    update_result = knowledge_service.check_for_updates(knowledge_dir, rag_system)
    print("✅ 知识库动态更新功能测试成功")
    print(f"  - 新文件数量: {len(update_result['new_files'])}")
    print(f"  - 更新文件数量: {len(update_result['updated_files'])}")
except Exception as e:
    print(f"❌ 知识库动态更新功能测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 5: 测试 RAGAS 评测功能
print("\n5. 测试 RAGAS 评测功能...")
try:
    rag_evaluation = RAGEvaluation()
    # 生成测试用例
    test_cases = rag_evaluation.generate_test_cases(rag_system.knowledge_cache, count=2)
    print("✅ 测试用例生成成功")
    print(f"  - 生成测试用例数量: {len(test_cases)}")
    
    if test_cases:
        # 执行评测（可选，可能需要较长时间）
        print("  - 评测功能已就绪，可通过 API 调用")
except Exception as e:
    print(f"❌ RAGAS 评测功能测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")