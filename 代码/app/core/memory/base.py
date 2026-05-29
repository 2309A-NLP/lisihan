# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""记忆存储抽象基类。

定义所有记忆存储必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseMemory(ABC):
    """记忆存储基类。"""

    @abstractmethod
    def connect(self) -> None:
        """连接到存储服务。"""
        pass

    @abstractmethod
    def save(self, user_id: int, role_id: int, message: Dict) -> bool:
        """保存消息到记忆。"""
        pass

    @abstractmethod
    def get(self, user_id: int, role_id: int, limit: int = 50) -> List[Dict]:
        """获取记忆。"""
        pass

    @abstractmethod
    def clear(self, user_id: int, role_id: int) -> None:
        """清除记忆。"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查存储是否可用。"""
        pass


class BaseLongTermMemory(BaseMemory):
    """长期记忆基类（支持向量检索）。"""

    @abstractmethod
    def search(
        self,
        user_id: int,
        role_id: int,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Dict]:
        """向量相似度搜索。"""
        pass

    @abstractmethod
    def count(self, user_id: int = None, role_id: int = None) -> int:
        """统计记忆数量。"""
        pass

    @abstractmethod
    def flush(self) -> bool:
        """刷盘持久化。"""
        pass