#!/usr/bin/env python3
"""
删除旧集合并重新创建新集合
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.vectorstore import vector_store

print("开始删除旧集合并重新创建新集合...")

# 连接 Milvus
if not vector_store.client:
    vector_store.connect()

# 删除旧集合
try:
    print("尝试删除旧集合: " + vector_store.collection_name)
    vector_store.client.drop_collection(vector_store.collection_name)
    print("成功删除旧集合")
except Exception as e:
    print("删除旧集合失败: " + str(e))

# 创建新集合
try:
    print("开始创建新集合...")
    vector_store.create_collection()
    print("成功创建新集合")
except Exception as e:
    print("创建新集合失败: " + str(e))
    import traceback
    traceback.print_exc()

print("\n操作完成！")