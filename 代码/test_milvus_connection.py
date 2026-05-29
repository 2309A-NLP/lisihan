#!/usr/bin/env python3
"""
测试 Milvus 连接和基本操作
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.vectorstore import vector_store
from app.core.rag import RAGSystem

print("开始测试 Milvus 连接...")

# 测试 1: 检查 Milvus 连接
print("\n1. 测试 Milvus 连接...")
if vector_store.client:
    print("Milvus 连接成功")
    print("  - 连接地址: " + vector_store.uri)
else:
    print("Milvus 连接失败")
    # 尝试重新连接
    print("  - 尝试重新连接...")
    vector_store.connect()
    if vector_store.client:
        print("重新连接成功")
    else:
        print("重新连接失败")

# 测试 2: 测试集合创建
print("\n2. 测试集合创建...")
try:
    vector_store.create_collection()
    print("集合创建测试成功")
except Exception as e:
    print("集合创建测试失败: " + str(e))

# 测试 3: 测试向量插入
print("\n3. 测试向量插入...")
try:
    # 创建 RAG 系统实例获取嵌入
    rag_system = RAGSystem()
    test_content = "测试 Milvus 向量插入"
    embedding = rag_system.get_embedding(test_content)
    
    if embedding:
        success = vector_store.insert(test_content, embedding)
        if success:
            print("向量插入成功")
        else:
            print("向量插入失败")
    else:
        print("获取嵌入失败")
except Exception as e:
    print("向量插入测试失败: " + str(e))
    import traceback
    traceback.print_exc()

# 测试 4: 测试向量搜索
print("\n4. 测试向量搜索...")
try:
    test_query = "测试"
    embedding = rag_system.get_embedding(test_query)
    
    if embedding:
        results = vector_store.search(embedding, top_k=3)
        print("向量搜索成功，返回结果数量: " + str(len(results)))
        for i, result in enumerate(results):
            print("  - 结果 " + str(i+1) + ": " + result['content'][:50] + "...")
    else:
        print("获取查询嵌入失败")
except Exception as e:
    print("向量搜索测试失败: " + str(e))
    import traceback
    traceback.print_exc()

print("\n测试完成！")