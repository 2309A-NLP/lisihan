# -*- coding: utf-8 -*-
from pymilvus import connections, Collection
import sys


# Function: Inspect Milvus knowledge data for debugging.
def check_milvus_data():
    # 连接 Milvus
    connections.connect(
        alias="default",
        host='192.168.190.128',
        port='19530'
    )

    # 检查集合
    collection_name = 'chatbot_knowledge'

    try:
        collection = Collection(collection_name)
        collection.load()

        # 获取数据量
        num_entities = collection.num_entities
        print(f"集合 '{collection_name}' 中的数据量: {num_entities}")

        if num_entities > 0:
            # 查看前5条数据
            results = collection.query(
                expr="id >= 0",
                output_fields=["id", "role_id", "content", "vector"],
                limit=5
            )
            print("\n前5条数据:")
            for i, result in enumerate(results, 1):
                print(f"\n第 {i} 条:")
                print(f"  ID: {result.get('id')}")
                print(f"  角色ID: {result.get('role_id')}")
                content = result.get('content', '')
                print(f"  内容: {content[:100]}..." if len(content) > 100 else f"  内容: {content}")
                if 'vector' in result:
                    print(f"  向量维度: {len(result['vector'])}")
        else:
            print(f"⚠️ 集合 '{collection_name}' 是空的！")
            print("知识库数据没有存入 Milvus")

    except Exception as e:
        print(f"错误: {e}")
        print("可能集合不存在或连接失败")

    # 列出所有集合
    from pymilvus import utility
    collections = utility.list_collections()
    print(f"\n所有集合: {collections}")


if __name__ == "__main__":
    check_milvus_data()