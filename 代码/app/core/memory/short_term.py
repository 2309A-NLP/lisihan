# -*- coding: utf-8 -*-
"""短期记忆管理器（Redis/内存缓存）。

负责存储最近对话，提供快速访问。
"""

import json
from typing import Dict, List, Optional

try:
    import redis
except ImportError:
    redis = None

from config.config import settings
from app.core.memory.base import BaseMemory


class ShortTermMemory(BaseMemory):
    """短期记忆管理器。

    功能：
    - 存储最近 N 条对话
    - 支持 Redis 和内存降级两种模式
    - 自动过期（24小时）
    """

    MAX_HISTORY_LENGTH = getattr(settings, "MAX_HISTORY_LENGTH", 50)

    def __init__(self):
        self._redis_client: Optional[redis.Redis] = None
        self._memory_storage: Dict[str, List[str]] = {}

    def connect(self) -> None:
        """连接到 Redis 服务器。"""
        if redis is None:
            print("未安装 redis，短期缓存将使用内存降级模式")
            return

        try:
            self._redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
            self._redis_client.ping()
            print("成功连接到 Redis")
        except Exception as e:
            print(f"连接 Redis 失败: {e}")
            self._redis_client = None

    def is_available(self) -> bool:
        """检查存储是否可用。"""
        if self._redis_client:
            try:
                self._redis_client.ping()
                return True
            except Exception:
                return False
        return bool(self._memory_storage is not None)

    def _get_key(self, user_id: int, role_id: int) -> str:
        """生成 Redis 键名。"""
        return f"chat:{user_id}:{role_id}"

    def get(self, user_id: int, role_id: int, limit: int = None) -> List[Dict]:
        """获取短期记忆。

        Args:
            user_id: 用户 ID
            role_id: 角色 ID
            limit: 返回数量限制（默认使用 MAX_HISTORY_LENGTH）

        Returns:
            消息列表
        """
        key = self._get_key(user_id, role_id)
        max_len = limit or self.MAX_HISTORY_LENGTH

        if not self._redis_client:
            messages = self._memory_storage.get(key, [])[:max_len]
            return [json.loads(msg) for msg in messages]

        try:
            messages = self._redis_client.lrange(key, 0, max_len - 1)
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            print(f"获取短期缓存失败: {e}")
            return []

    def save(self, user_id: int, role_id: int, message: Dict) -> bool:
        """保存短期记忆。

        Args:
            user_id: 用户 ID
            role_id: 角色 ID
            message: 消息对象

        Returns:
            是否保存成功
        """
        key = self._get_key(user_id, role_id)
        payload = json.dumps(message, ensure_ascii=False)

        if not self._redis_client:
            self._memory_storage.setdefault(key, [])
            self._memory_storage[key].insert(0, payload)
            self._memory_storage[key] = self._memory_storage[key][:self.MAX_HISTORY_LENGTH]
            return True

        try:
            self._redis_client.lpush(key, payload)
            self._redis_client.ltrim(key, 0, self.MAX_HISTORY_LENGTH - 1)
            self._redis_client.expire(key, 86400)  # 24小时
            return True
        except Exception as e:
            print(f"保存短期缓存失败: {e}")
            return False

    def clear(self, user_id: int, role_id: int) -> None:
        """清除短期记忆。"""
        key = self._get_key(user_id, role_id)

        if not self._redis_client:
            self._memory_storage.pop(key, None)
            return

        try:
            self._redis_client.delete(key)
        except Exception as e:
            print(f"清空短期缓存失败: {e}")

    def get_raw_client(self):
        """获取原始 Redis 客户端（用于特殊操作）。"""
        return self._redis_client