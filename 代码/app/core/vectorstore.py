"""
向量数据库管理模块
==================
这个模块封装了对Milvus向量数据库的操作，提供：
- 连接到Milvus服务器
- 创建和管理集合（Collection）
- 插入向量数据
- 向量相似度搜索

向量数据库的作用：
- 存储知识片段的向量表示（768维）
- 通过向量相似度快速检索相关内容
- 支持语义搜索（不只是关键词匹配）

与知识库的关系：
- knowledge/data 中的文件 → 解析 → 文本分块 → 向量化 → 存入Milvus
- 用户查询 → 向量化 → 在Milvus中搜索 → 返回最相关的知识片段
"""

import logging

# ========== PyMilvus 导入（带降级处理）==========
try:
    # 导入Milvus客户端和数据类型
    from pymilvus import MilvusClient, DataType

    # MilvusClient: Milvus数据库客户端，提供所有操作接口
    # DataType: 定义字段类型（INT64, VARCHAR, FLOAT_VECTOR等）
    PYMILVUS_IMPORT_ERROR = None  # 无错误标记
except Exception as exc:
    # 如果导入失败（未安装pymilvus），设为None并记录错误
    MilvusClient = None
    DataType = None
    PYMILVUS_IMPORT_ERROR = exc  # 保存错误信息，方便调试

# 导入项目配置
from config.config import settings

logger = logging.getLogger(__name__)
import numpy as np  # 数值计算，用于处理向量

class VectorStore:
    """
    向量数据库存储类

    封装Milvus操作，提供：
    - 连接管理
    - 集合（表）管理
    - 向量插入
    - 向量搜索

    使用场景：
    1. 知识库文件处理时：将每个文本块的向量存入
    2. 用户查询时：将问题向量化，搜索相关知识

    注意：
    - 向量维度固定为768（对应Sentence-BERT等模型）
    - 使用L2距离（欧氏距离）衡量相似度
    - 索引类型IVF_FLAT（平衡速度和精度）
    """

    def __init__(self):
        """
        初始化向量存储

        从配置读取Milvus连接参数
        """
        # 构建Milvus连接URI
        self.uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"

        # 集合名称（从配置读取）
        # 这个集合存储知识库的向量数据
        self.collection_name = settings.MILVUS_COLLECTION

        # Milvus客户端实例
        self.client = None

        # 连接由 init_milvus() 在 FastAPI startup 中触发，避免模块导入时重复初始化。

    def connect(self):
        """
        连接到Milvus服务器

        功能：
        - 创建Milvus客户端
        - 测试连接是否成功
        - 连接成功后创建或加载集合

        注意：
        - 如果pymilvus未安装，会降级处理（client=None）
        - 连接失败时不会抛出异常，而是设置client=None
        """
        try:
            # 检查pymilvus是否可用
            if MilvusClient is None:
                print(f"pymilvus 不可用，向量数据库将使用离线降级模式。原因: {PYMILVUS_IMPORT_ERROR}")
                self.client = None
                return

            # 创建Milvus客户端
            # MilvusClient会自动处理连接池和重连
            self.client = MilvusClient(uri=self.uri, timeout=5)
            print("成功连接到Milvus: " + self.uri)

            # 连接成功后创建或验证集合
            self.create_collection()

        except Exception as e:
            print("连接Milvus失败: " + str(e))

    def create_collection(self):
        """
        创建向量集合（相当于关系数据库的表）

        Milvus集合需要定义：
        1. Schema（字段定义）：id, content, embedding
        2. 索引（加速搜索）：对embedding字段建索引

        集合结构：
        - id: 主键，自动生成
        - content: 文本内容（原始知识片段）
        - embedding: 向量（768维浮点数）

        注意：
        - 如果集合已存在，直接使用
        - 如果创建失败，会打印错误信息
        """
        try:
            # 检查客户端是否可用
            if not self.client:
                print("客户端未初始化，跳过创建集合")
                return

            # ========== 检查集合是否存在 ==========
            print("检查集合是否存在: " + self.collection_name)
            has_collection = False
            try:
                # has_collection() 返回布尔值，判断集合是否存在
                has_collection = self.client.has_collection(self.collection_name)
                print("检查集合存在结果: " + str(has_collection))
            except Exception as e:
                print("检查集合存在失败: " + str(e))

            # 如果集合已存在，直接返回
            if has_collection:
                print("集合 " + self.collection_name + " 已存在")
                return

            # ========== 清理可能存在的旧集合 ==========
            # 考虑到之前的创建可能失败，先尝试删除
            try:
                print("尝试删除旧集合: " + self.collection_name)
                self.client.drop_collection(self.collection_name)
                print("已删除旧集合: " + self.collection_name)
            except Exception as e:
                # 删除失败可能因为集合不存在，忽略错误
                print("删除旧集合失败: " + str(e))

            # ========== 创建新集合 ==========
            print("开始创建集合: " + self.collection_name)

            # 步骤1: 创建Schema（数据结构定义）
            # auto_id=True: ID自动生成（无需手动指定）
            # enable_dynamic_field=True: 允许动态添加字段（扩展性）
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            print("创建 schema 成功")

            # 步骤2: 添加字段
            # 主键ID（INT64类型）
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

            # 文本内容（VARCHAR类型，最大65535字符）
            # 存储原始知识片段文本
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

            # 向量字段（FLOAT_VECTOR类型，768维）
            # 存储文本对应的向量表示
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
            print("添加字段成功")

            # 步骤3: 创建索引（加速搜索）
            # 索引就像书的目录，没有索引需要全表扫描
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",  # 对向量字段建索引
                index_type="IVF_FLAT",  # 索引类型：IVF_FLAT（平衡速度和精度）
                metric_type="L2",  # 距离度量：L2欧氏距离（越小越相似）
                params={"nlist": 1024}  # 索引参数：聚类中心数量
            )
            print("准备索引参数成功")

            # 步骤4: 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            print("成功创建集合: " + self.collection_name)

            # 步骤5: 验证集合是否创建成功
            try:
                has_collection = self.client.has_collection(self.collection_name)
                print("验证集合创建结果: " + str(has_collection))
            except Exception as e:
                print("验证集合创建失败: " + str(e))

        except Exception as e:
            print("创建集合失败: " + str(e))
            logger.exception("创建 Milvus 集合失败: collection=%s, error=%s", self.collection_name, e)

    def insert(self, content: str, embedding: list, role_ids: list = None, source_file: str = "") -> bool:
        """
        插入向量到Milvus

        用于将处理后的知识片段存入向量数据库

        参数：
            content: 文本内容（原始知识片段）
            embedding: 向量表示（768维浮点数列表）

        返回：
            bool: 是否插入成功

        使用示例：
            vector_store.insert(
                content="民法典第1043条规定...",
                embedding=[0.123, 0.456, ...]  # 768个浮点数
            )
        """
        try:
            # 检查客户端是否可用
            if not self.client:
                print("客户端未初始化，尝试重新连接")
                self.connect()  # 尝试重新连接
                if not self.client:
                    print("重新连接失败，跳过插入")
                    return False

            # 准备插入数据
            # 按照集合的Schema组织数据
            data = [
                {
                    "content": content,  # 文本内容
                    "embedding": embedding,  # 向量表示
                    "role_ids": [int(role_id) for role_id in (role_ids or [])],
                    "source_file": source_file or "",
                }
            ]

            # 执行插入
            # insert方法返回插入结果（包含生成的ID）
            self.client.insert(collection_name=self.collection_name, data=data)
            return True

        except Exception as e:
            print(f"插入向量失败: {e}")
            return False

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        """
        搜索相似向量（核心功能）

        在Milvus中查找与查询向量最相似的K个向量

        工作原理：
        1. 将查询向量与所有存储向量计算距离
        2. 使用索引加速，不需要全量计算
        3. 返回距离最小的K个结果

        参数：
            query_embedding: 查询向量（768维）
            top_k: 返回相似结果的数量（默认5）

        返回：
            list: 搜索结果，每个结果包含：
                - content: 文本内容
                - distance: 欧氏距离（越小越相似）

        使用示例：
            # 假设用户输入"什么是民法典"
            query_vec = embedding_model.encode("什么是民法典")
            results = vector_store.search(query_vec, top_k=3)
            for r in results:
                print(f"相似度: {r['distance']}, 内容: {r['content']}")
        """
        try:
            # 检查客户端是否可用
            if not self.client:
                print("客户端未初始化，尝试重新连接")
                self.connect()
                if not self.client:
                    print("重新连接失败，返回空结果")
                    return []

            # 搜索参数配置
            search_params = {
                "metric_type": "L2",  # 使用L2欧氏距离
                "params": {"nprobe": 10}  # nprobe: 搜索时访问的聚类数量（越大越精确但越慢）
            }

            # 执行向量搜索
            # data: 查询向量列表（支持批量搜索）
            # limit: 返回数量
            # output_fields: 需要返回的字段
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],  # 查询向量
                limit=top_k,  # 最多返回top_k条
                search_params=search_params,  # 搜索参数
                output_fields=["content", "role_ids", "source_file"]
            )

            # 处理搜索结果
            # results结构: [[hit1, hit2, ...]] (批次数 × 每批结果)
            search_results = []
            for hits in results:  # 遍历每一批结果
                for hit in hits:  # 遍历每批中的每个结果
                    search_results.append({
                        "content": hit.get("entity", {}).get("content", ""),  # 提取文本内容
                        "role_ids": hit.get("entity", {}).get("role_ids", []),
                        "source_file": hit.get("entity", {}).get("source_file", ""),
                        "distance": hit.get("distance", 0.0)  # 获取距离值
                    })

            return search_results

        except Exception as e:
            print(f"搜索向量失败: {e}")
            return []  # 搜索失败返回空列表

    def get_collection(self):
        """
        获取集合信息

        返回：
            str: 集合名称（如果存在），否则None
        """
        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                return self.collection_name
            return None
        except Exception as e:
            print(f"获取集合失败: {e}")
            return None

    def count(self, collection_name: str = None) -> int:
        """返回指定 Milvus 集合中的实体数量，默认统计知识库集合。"""
        try:
            if not self.client:
                self.connect()
            if not self.client:
                return 0

            target_collection = collection_name or self.collection_name
            if not self.client.has_collection(target_collection):
                return 0

            self.flush(target_collection)

            try:
                stats = self.client.get_collection_stats(target_collection)
                if isinstance(stats, dict) and stats.get("row_count") is not None:
                    return int(stats["row_count"])
            except Exception:
                pass

            rows = self.client.query(
                collection_name=target_collection,
                filter="id >= 0",
                output_fields=["count(*)"],
            )
            if rows and rows[0].get("count(*)") is not None:
                return int(rows[0]["count(*)"])
            return 0
        except Exception as e:
            print(f"获取Milvus数据量失败: {e}")
            return 0

    def flush(self, collection_name: str = None) -> bool:
        """把待写入数据刷盘，避免刚插入后可视化页面仍显示 0。"""
        try:
            if not self.client:
                return False
            target_collection = collection_name or self.collection_name
            if self.client.has_collection(target_collection):
                self.client.flush(target_collection)
                return True
        except Exception as e:
            print(f"刷新Milvus集合失败: {e}")
        return False

    def insert_many(self, items: list, embedding_fn) -> int:
        """批量写入知识片段到 Milvus，返回成功写入数量。"""
        inserted = 0
        for item in items:
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            embedding = embedding_fn(content)
            if not embedding:
                continue
            if self.insert(
                content,
                embedding,
                role_ids=item.get("role_ids", []),
                source_file=item.get("source_file", ""),
            ):
                inserted += 1
        if inserted:
            self.flush()
        return inserted

    def insert_missing(self, items: list, embedding_fn) -> int:
        """把本地知识缓存中 Milvus 还没有的片段补写进去，避免新增内置知识只存在本地。"""
        try:
            if not self.client:
                self.connect()
            if not self.client or not self.client.has_collection(self.collection_name):
                return 0

            current_count = self.count()
            existing_rows = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["content"],
                limit=max(current_count + 100, 1000),
            )
            existing_contents = {str(row.get("content", "") or "").strip() for row in existing_rows}
            missing_items = [
                item for item in items
                if str(item.get("content", "") or "").strip()
                and str(item.get("content", "") or "").strip() not in existing_contents
            ]
            return self.insert_many(missing_items, embedding_fn)
        except Exception as e:
            print(f"同步缺失知识到Milvus失败: {e}")
            return 0

    def rebuild_from_items(self, items: list, embedding_fn) -> int:
        """清空并重建知识库集合，用当前 embedding 模型重新生成全部向量。"""
        try:
            if not self.client:
                self.connect()
            if not self.client:
                return 0

            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
            self.create_collection()
            return self.insert_many(items, embedding_fn)
        except Exception as e:
            print(f"重建Milvus知识库失败: {e}")
            return 0


# ========== 全局单例 ==========
# 创建全局的VectorStore实例，整个应用共享
# 这样可以避免重复连接Milvus
vector_store = VectorStore()


def init_milvus():
    """
    初始化Milvus（应用启动时调用）

    功能：
    - 确保Milvus连接正常
    - 确保集合存在
    - 可以在这里添加额外的初始化逻辑

    调用时机：
    - 应用启动时自动调用
    - 可以在配置热重载时重新调用
    """
    # 确保连接
    vector_store.connect()

    # 如果连接失败，直接返回
    if not vector_store.client:
        return

    # 检查集合是否存在（带异常处理）
    has_collection = False
    try:
        has_collection = vector_store.client.has_collection(settings.MILVUS_COLLECTION)
    except Exception as e:
        print("检查集合存在失败: " + str(e))

    # 如果客户端可用但集合不存在，创建集合
    if vector_store.client and not has_collection:
        vector_store.create_collection()
