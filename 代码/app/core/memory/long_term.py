# -*- coding: utf-8 -*-
"""长期记忆管理器（Milvus 向量数据库）。

负责存储和检索历史对话，支持语义搜索。
"""

import hashlib
import time
from typing import Dict, List, Optional

try:
    from pymilvus import DataType, MilvusClient
except ImportError:
    DataType = None
    MilvusClient = None

from config.config import settings
from app.core.memory.base import BaseLongTermMemory


class LongTermMemory(BaseLongTermMemory):
    """长期记忆管理器。

    功能：
    - 持久化存储对话历史
    - 向量相似度检索
    - 按时间排序查询
    """

    VECTOR_DIM = 768

    def __init__(self):
        self._client: Optional[MilvusClient] = None
        self._collection_name = settings.MILVUS_MEMORY_COLLECTION

    def connect(self) -> None:
        """连接到 Milvus 服务器。"""
        if MilvusClient is None:
            print("pymilvus 不可用，长期记忆功能不可用")
            return

        try:
            uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            self._client = MilvusClient(uri=uri, timeout=5)
            self._ensure_collection()
            print(f"成功连接到 Milvus 长期记忆集合: {self._collection_name}")
        except Exception as e:
            print(f"连接 Milvus 失败: {e}")
            self._client = None

    def is_available(self) -> bool:
        """检查存储是否可用。"""
        if not self._client:
            return False
        try:
            return self._client.has_collection(self._collection_name)
        except Exception:
            return False

    @property
    def milvus_client(self):
        """兼容旧代码访问方式。"""
        return self._client

    def _ensure_collection(self) -> None:
        """确保集合存在，不存在则创建。"""
        if not self._client:
            return

        if self._client.has_collection(self._collection_name):
            self._load_collection()
            return

        # 创建 schema
        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="user_id", datatype=DataType.INT64)
        schema.add_field(field_name="role_id", datatype=DataType.INT64)
        schema.add_field(field_name="conversation_id", datatype=DataType.INT64)
        schema.add_field(field_name="sender", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="timestamp", datatype=DataType.INT64)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.VECTOR_DIM)

        # 创建索引
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128},
        )

        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )

        self._load_collection()

    def _load_collection(self) -> None:
        """加载集合到内存。"""
        try:
            self._client.load_collection(self._collection_name)
        except Exception:
            pass

    def _build_embedding(self, text: str) -> List[float]:
        """构建向量表示（简化版，实际应用应使用专业 embedding 模型）。"""
        if not text:
            return [0.0] * self.VECTOR_DIM

        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in seed]
        repeats = (self.VECTOR_DIM // len(values)) + 1
        return (values * repeats)[:self.VECTOR_DIM]

    def save(
        self,
        user_id: int,
        role_id: int,
        conversation_id: int = None,
        message: Dict = None
    ) -> bool:
        """保存长期记忆。

        注意：此方法签名与基类略有不同，增加了 conversation_id。

        Args:
            user_id: 用户 ID
            role_id: 角色 ID
            conversation_id: 会话 ID
            message: 消息对象

        Returns:
            是否保存成功
        """
        if not self._client:
            self.connect()
            if not self._client:
                return False

        if message is None:
            return False

        content = str(message.get("content", "") or "")
        if not content.strip():
            return False

        data = [{
            "user_id": int(user_id),
            "role_id": int(role_id),
            "conversation_id": int(conversation_id) if conversation_id else 0,
            "sender": str(message.get("sender", ""))[:20],
            "content": content[:65535],
            "timestamp": int(message.get("timestamp") or time.time()),
            "embedding": self._build_embedding(content),
        }]

        try:
            self._client.insert(collection_name=self._collection_name, data=data)
            self.flush()
            return True
        except Exception as e:
            print(f"保存长期记忆失败: {e}")
            return False

    def get(
        self,
        user_id: int,
        role_id: int,
        limit: int = 50,
        conversation_id: int = None
    ) -> List[Dict]:
        """获取长期记忆（按时间排序）。

        Args:
            user_id: 用户 ID
            role_id: 角色 ID
            limit: 返回数量
            conversation_id: 可选，指定会话 ID

        Returns:
            消息列表
        """
        if not self._client:
            self.connect()
        if not self._client:
            return []

        # 构建过滤条件
        filter_str = f"user_id == {int(user_id)} and role_id == {int(role_id)}"
        if conversation_id is not None:
            filter_str += f" and conversation_id == {int(conversation_id)}"

        try:
            rows = self._client.query(
                collection_name=self._collection_name,
                filter=filter_str,
                output_fields=["user_id", "role_id", "conversation_id", "sender", "content", "timestamp"],
                limit=min(limit, 1000),
            )
            rows = sorted(rows, key=lambda item: item.get("timestamp", 0), reverse=True)
            return rows[:limit]
        except Exception as e:
            print(f"获取长期记忆失败: {e}")
            return []

    def get_by_conversation(self, conversation_id: int, limit: int = 50) -> List[Dict]:
        """按会话获取长期记忆。"""
        if not self._client:
            self.connect()
        if not self._client:
            return []

        try:
            rows = self._client.query(
                collection_name=self._collection_name,
                filter=f"conversation_id == {int(conversation_id)}",
                output_fields=["user_id", "role_id", "conversation_id", "sender", "content", "timestamp"],
                limit=min(limit, 1000),
            )
            rows = sorted(rows, key=lambda item: item.get("timestamp", 0))
            return rows
        except Exception as e:
            print(f"按会话获取长期记忆失败: {e}")
            return []

    def search(
        self,
        user_id: int,
        role_id: int,
        query_vector: List[float],
        limit: int = 10,
        conversation_id: int = None
    ) -> List[Dict]:
        """向量相似度搜索。"""
        if not self._client:
            self.connect()
        if not self._client or not query_vector:
            return []

        if len(query_vector) != self.VECTOR_DIM:
            query_vector = (query_vector + [0.0] * self.VECTOR_DIM)[:self.VECTOR_DIM]

        # 构建过滤条件
        filter_str = f"user_id == {int(user_id)} and role_id == {int(role_id)}"
        if conversation_id is not None:
            filter_str += f" and conversation_id == {int(conversation_id)}"

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                data=[query_vector],
                limit=limit,
                filter=filter_str,
                output_fields=["user_id", "role_id", "conversation_id", "sender", "content", "timestamp"],
                search_params={"metric_type": "L2", "params": {"nprobe": 10}},
            )

            memory = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
                    memory.append({
                        "user_id": entity.get("user_id"),
                        "role_id": entity.get("role_id"),
                        "conversation_id": entity.get("conversation_id"),
                        "sender": entity.get("sender"),
                        "content": entity.get("content"),
                        "timestamp": entity.get("timestamp"),
                        "distance": hit.get("distance", 0.0) if isinstance(hit, dict) else 0.0,
                    })
            return memory
        except Exception as e:
            print(f"搜索长期记忆失败: {e}")
            return []

    def count(self, user_id: int = None, role_id: int = None) -> int:
        """统计记录数量。

        Args:
            user_id: 可选，按用户过滤
            role_id: 可选，按角色过滤

        Returns:
            记录数量
        """
        if not self._client:
            self.connect()
        if not self._client or not self._client.has_collection(self._collection_name):
            return 0

        try:
            self.flush()

            filter_str = ""
            if user_id is not None and role_id is not None:
                filter_str = f"user_id == {int(user_id)} and role_id == {int(role_id)}"
            elif user_id is not None:
                filter_str = f"user_id == {int(user_id)}"
            elif role_id is not None:
                filter_str = f"role_id == {int(role_id)}"

            if filter_str:
                total = 0
                iterator = self._client.query_iterator(
                    collection_name=self._collection_name,
                    batch_size=1000,
                    limit=-1,
                    filter=filter_str,
                    output_fields=["id"],
                )
                try:
                    while True:
                        batch = iterator.next()
                        if not batch:
                            break
                        total += len(batch)
                finally:
                    iterator.close()
                return total

            stats = self._client.get_collection_stats(self._collection_name)
            if isinstance(stats, dict) and stats.get("row_count") is not None:
                return int(stats["row_count"])

            return 0
        except Exception as e:
            print(f"统计长期记忆数量失败: {e}")
            return 0

    def clear(self, user_id: int, role_id: int) -> None:
        """清除指定用户的长期记忆（谨慎使用）。"""
        if not self._client:
            self.connect()
        if not self._client:
            return

        try:
            filter_str = f"user_id == {int(user_id)} and role_id == {int(role_id)}"
            self._client.delete(
                collection_name=self._collection_name,
                filter=filter_str,
            )
        except Exception as e:
            print(f"清除长期记忆失败: {e}")

    def flush(self) -> bool:
        """刷盘，确保数据持久化。"""
        if not self._client:
            return False
        try:
            if self._client.has_collection(self._collection_name):
                self._client.flush(collection_name=self._collection_name)
                return True
        except Exception:
            pass
        return False
