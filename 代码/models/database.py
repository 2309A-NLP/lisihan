# -*- coding: utf-8 -*-
"""数据库连接与初始化。

这个模块只负责 SQLAlchemy 基础设施：
- Base：模型声明基类
- engine：数据库引擎
- SessionLocal/db：会话工厂
- initialize_database()：应用启动时显式建库/建表
"""

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from config.config import settings


def _ensure_mysql_database(database_url: str):
    """当目标 MySQL 数据库不存在时先创建数据库。"""
    if not database_url.startswith("mysql"):
        return
    url = make_url(database_url)
    database = url.database
    if not database:
        return
    server_url = url.set(database=None)
    server_engine = create_engine(
        server_url,
        connect_args={"use_pure": True, "connection_timeout": 5},
        pool_pre_ping=True,
    )
    with server_engine.connect() as connection:
        connection.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{database}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        connection.commit()
    server_engine.dispose()


if not settings.DATABASE_URL:
    raise RuntimeError(
        "MySQL 配置未完成：请填写 config/mysql_config.py 后再启动项目。"
    )

try:
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"use_pure": True, "connection_timeout": 5},
        pool_pre_ping=True,
    )
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "缺少 MySQL 驱动，请先运行：.\\venv\\Scripts\\python.exe -m pip install mysql-connector-python"
    ) from exc
except Exception as exc:
    raise RuntimeError(
        f"MySQL 引擎创建失败，请检查数据库配置。原始错误: {exc}"
    ) from exc


SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
db = SessionLocal


def get_db():
    """为 FastAPI 依赖提供数据库会话。"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        SessionLocal.remove()


def initialize_database() -> None:
    """应用启动时显式初始化数据库。"""
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        message = str(exc).lower()
        if "unknown database" not in message and "1049" not in message:
            raise
        _ensure_mysql_database(settings.DATABASE_URL)
        Base.metadata.create_all(bind=engine)
