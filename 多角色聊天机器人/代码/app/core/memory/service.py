# -*- coding: utf-8 -*-
"""记忆服务统一入口。

外部代码只需要使用全局单例 ``memory``：
- ``memory.short`` 访问短期记忆（Redis/内存）
- ``memory.long`` 访问长期记忆（Milvus）
- ``memory.connect()`` 在应用启动时连接两类存储
"""

from app.core.memory.long_term import LongTermMemory
from app.core.memory.short_term import ShortTermMemory


class MemoryService:
    """组合短期记忆和长期记忆的应用级服务。"""

    def __init__(self):
        self._short = None
        self._long = None

    @property
    def short(self) -> ShortTermMemory:
        """短期记忆存储。"""
        if self._short is None:
            self._short = ShortTermMemory()
            self._short.connect()
        return self._short

    @property
    def long(self) -> LongTermMemory:
        """长期记忆存储。"""
        if self._long is None:
            self._long = LongTermMemory()
            self._long.connect()
        return self._long

    def connect(self) -> None:
        """连接所有记忆存储。"""
        self.short.connect()
        self.long.connect()

    def is_ready(self) -> dict:
        """返回记忆服务可用状态。"""
        return {
            "short": self.short.is_available(),
            "long": self.long.is_available(),
        }


memory = MemoryService()


def initialize_memory() -> None:
    """应用启动时初始化记忆服务。"""
    memory.connect()
