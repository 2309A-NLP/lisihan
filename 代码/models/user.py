"""
用户数据模型模块
================
这个模块定义了用户表的数据库结构，用于管理用户账号信息。

用户表功能：
- 存储用户账号信息（用户名、密码、邮箱）
- 用户认证和授权
- 关联用户的对话历史和记忆

安全设计：
- 密码使用哈希存储（不存明文）
- 用户名和邮箱唯一性约束
- 自动记录创建和更新时间

与对话的关联：
users (用户表) 1 ── N conversations (对话表)
一个用户可以拥有多个对话会话
"""

# ========== SQLAlchemy 导入 ==========
from sqlalchemy import Column, Integer, String, DateTime, Boolean  # ✅ 添加 Boolean
# Column: 定义数据库表的列（字段）
# Integer: 整数类型（用于ID）
# String: 字符串类型（用于用户名、密码、邮箱）
# DateTime: 日期时间类型（用于时间戳）
# Boolean: 布尔类型（用于 is_active, is_admin）

from sqlalchemy.sql import func
# func.now(): 获取当前时间的SQL函数

from app.models.database import Base


# Function: Map the users database table.
class User(Base):
    """
    用户表模型
    """

    __tablename__ = "users"

    # ========== 基础字段 ==========
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # ========== 新增字段（用户状态控制）==========
    is_active = Column(Boolean, default=True)                    # 账号是否启用
    is_admin = Column(Boolean, default=False)                    # 是否管理员
    last_login = Column(DateTime(timezone=True))                 # 最后登录时间 ✅ 添加 timezone
    login_count = Column(Integer, default=0)                     # 登录次数
