# -*- coding: utf-8 -*-
"""记忆管理模块。

统一入口：
    from app.core.memory import memory, initialize_memory

    initialize_memory()
    memory.short.save(user_id, role_id, message)
    memory.long.save(user_id, role_id, conversation_id, message)
    memory.long.search(user_id, role_id, query_embedding)
"""

from app.core.memory.service import MemoryService, initialize_memory, memory
from app.core.memory.short_term import ShortTermMemory
from app.core.memory.long_term import LongTermMemory
from app.core.memory.base import BaseMemory, BaseLongTermMemory

__all__ = [
    "MemoryService",
    "memory",
    "initialize_memory",
    "ShortTermMemory",
    "LongTermMemory",
    "BaseMemory",
    "BaseLongTermMemory",
]
