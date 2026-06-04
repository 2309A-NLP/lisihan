#!/usr/bin/env python3
"""
测试 Milvus 连接和集合创建
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.vectorstore import vector_store

print("开始测试 Milvus 连接和集合创建...")

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
    import traceback
    traceback.print_exc()

# 测试 3: 验证集合是否存在
print("\n3. 验证集合是否存在...")
try:
    has_collection = vector_store.client.has_collection(vector_store.collection_name)
    print("集合存在: " + str(has_collection))
except Exception as e:
    print("验证集合存在失败: " + str(e))

print("\n测试完成！")