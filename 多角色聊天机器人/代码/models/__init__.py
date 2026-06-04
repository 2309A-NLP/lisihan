# -*- coding: utf-8 -*-
"""数据库模型导出入口。"""

from app.models.database import Base, SessionLocal, db, engine, get_db, initialize_database

# 导入所有模型，确保它们注册到 Base.metadata。
from app.models.user import User
from app.models.role import KnowledgeBase, Role, Template
from app.models.chat import ChatMessage, Conversation

__all__ = [
    "Base",
    "SessionLocal",
    "db",
    "engine",
    "get_db",
    "initialize_database",
    "User",
    "Role",
    "Template",
    "KnowledgeBase",
    "Conversation",
    "ChatMessage",
]
