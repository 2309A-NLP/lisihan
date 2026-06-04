"""
数据库模型定义模块
==================
这个模块定义了聊天系统的数据库表结构，使用SQLAlchemy ORM。

两张核心表：
1. conversations（对话表）：存储对话会话信息
2. chat_messages（聊天消息表）：存储具体的消息内容

表关系：
users (用户表) 1 ── N conversations (对话表) 1 ── N chat_messages (消息表)

数据流向：
用户聊天 → 创建/更新对话 → 保存消息 → 同时同步到Redis(短期)和Milvus(长期)
"""

# ========== SQLAlchemy 导入 ==========
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
# Column: 定义数据库表的列（字段）
# DateTime: 日期时间类型
# ForeignKey: 外键约束，用于表之间的关联
# Integer: 整数类型
# String: 字符串类型（有长度限制）
# Text: 文本类型（无长度限制，适合长文本）

from sqlalchemy.sql import func
# func: SQL函数工具，这里用于获取当前时间（func.now()）

# ========== 导入基础模型类 ==========
from app.models.database import Base


# Base: SQLAlchemy 的声明性基类
# 所有模型类都需要继承 Base，这样SQLAlchemy才能识别并创建表


# Function: Map the conversations database table.
class Conversation(Base):
    """
    对话表模型

    功能：
    - 记录用户与角色的每次对话会话
    - 一个对话可以包含多条消息
    - 支持多轮对话的上下文管理

    表名：conversations

    使用场景：
    - 用户点击"新建对话"时创建新记录
    - 用户发送消息时更新 updated_at
    - 前端展示对话列表时查询此表

    与用户的关系：
    - 一个用户可以拥有多个对话
    - 一个对话只能属于一个用户

    与角色的关系：
    - 一个对话只能对应一个角色
    - 不同角色的对话分开管理
    """

    # 表名定义
    __tablename__ = "conversations"
    # 在MySQL中将会创建名为 'conversations' 的表

    # ========== 字段定义 ==========

    # 主键ID
    # Integer: 整数类型
    # primary_key=True: 设为主键（唯一标识）
    # index=True: 创建索引，加速查询
    id = Column(Integer, primary_key=True, index=True)

    # 用户ID（外键）
    # ForeignKey("users.id"): 关联到 users 表的 id 字段
    # index=True: 创建索引，加速按用户查询
    # nullable=False: 不能为空
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    # 角色ID
    # index=True: 创建索引，加速按角色查询
    # nullable=False: 不能为空
    role_id = Column(Integer, index=True, nullable=False)

    # 对话标题
    # String(120): 字符串类型，最大120个字符
    # nullable=False: 不能为空
    # default="新对话": 默认值为"新对话"
    title = Column(String(120), nullable=False, default="新对话")

    # 创建时间
    # DateTime(timezone=True): 带时区的日期时间
    # server_default=func.now(): 数据库服务器默认值为当前时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 更新时间
    # server_default=func.now(): 创建时默认为当前时间
    # onupdate=func.now(): 更新时自动更新为当前时间
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Function: Map the chat_messages database table.
class ChatMessage(Base):
    """
    聊天消息表模型

    功能：
    - 存储用户和AI之间的具体消息内容
    - 每条消息关联到特定的对话
    - 按时间顺序排列形成对话历史

    表名：chat_messages

    使用场景：
    - 用户发送消息时保存
    - AI回复时保存
    - 前端加载历史记录时查询
    - RAG系统获取对话上下文

    消息流向：
    MySQL (chat_messages) → UI显示历史记录
    Redis → 短期记忆（快速访问）
    Milvus → 长期记忆（向量检索）

    注意：
    - MySQL主要负责UI显示，不用于RAG检索
    - 长期记忆存储是Milvus做的
    - 短期记忆缓存是Redis做的
    """

    # 表名定义
    __tablename__ = "chat_messages"
    # 在MySQL中将会创建名为 'chat_messages' 的表

    # ========== 字段定义 ==========

    # 主键ID
    # Integer: 整数类型
    # primary_key=True: 设为主键
    # index=True: 创建索引
    id = Column(Integer, primary_key=True, index=True)

    # 对话ID（外键）
    # ForeignKey("conversations.id"): 关联到 conversations 表的 id 字段
    # index=True: 创建索引，加速按对话查询
    # nullable=False: 不能为空
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True, nullable=False)

    # 用户ID（外键）
    # ForeignKey("users.id"): 关联到 users 表的 id 字段
    # index=True: 创建索引，加速按用户查询
    # nullable=False: 不能为空
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    # 角色ID
    # index=True: 创建索引，加速按角色查询
    # nullable=False: 不能为空
    role_id = Column(Integer, index=True, nullable=False)

    # 发送者标识
    # String(20): 最大20个字符
    # nullable=False: 不能为空
    # 可能的值: "user"（用户发送） 或 "role"（AI发送）
    sender = Column(String(20), nullable=False)

    # 消息内容
    # Text: 文本类型，无长度限制
    # nullable=False: 不能为空
    # 可以存储很长的消息内容
    content = Column(Text, nullable=False)

    # 创建时间
    # DateTime(timezone=True): 带时区的日期时间
    # server_default=func.now(): 数据库服务器默认值为当前时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())

