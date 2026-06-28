# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: MySQL 连接管理和数据库初始化
"""

from pathlib import Path
from typing import Optional

import pymysql
from pymysql.connections import Connection

from config import Config


class MySQLConnector:
    """负责创建 MySQL 连接并执行初始化 SQL。"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config

    def get_connection(self, use_database: bool = True) -> Connection:
        kwargs = {
            "host": self.config.MYSQL_HOST,
            "port": self.config.MYSQL_PORT,
            "user": self.config.MYSQL_USER,
            "password": self.config.MYSQL_PASSWORD,
            "charset": self.config.MYSQL_CHARSET,
            "cursorclass": pymysql.cursors.DictCursor,
            "autocommit": False,
        }
        if use_database:
            kwargs["database"] = self.config.MYSQL_DATABASE
        return pymysql.connect(**kwargs)

    def init_database(self) -> None:
        schema_path = Path(__file__).resolve().parent / "schema.sql"
        statements = self._split_sql(schema_path.read_text(encoding="utf-8"))
        connection = self.get_connection(use_database=False)
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        self._run_lightweight_migrations()

    def _run_lightweight_migrations(self) -> None:
        """兼容旧版本表结构，避免 CREATE TABLE IF NOT EXISTS 无法补字段。"""
        migrations = [
            """
            ALTER TABLE reminder_logs
            ADD COLUMN occurrence_time DATETIME NULL COMMENT '本次应提醒的发生时间'
            AFTER schedule_id
            """,
            """
            ALTER TABLE reminder_logs
            ADD COLUMN reminder_content VARCHAR(255) DEFAULT NULL COMMENT '提醒内容'
            AFTER occurrence_time
            """,
            """
            ALTER TABLE reminder_logs
            ADD UNIQUE KEY uk_schedule_occurrence (schedule_id, occurrence_time)
            """,
        ]
        connection = self.get_connection(use_database=True)
        try:
            with connection.cursor() as cursor:
                for statement in migrations:
                    try:
                        cursor.execute(statement)
                    except pymysql.err.OperationalError as exc:
                        if exc.args and exc.args[0] in {1060, 1061}:
                            continue
                        raise
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _split_sql(sql_text: str) -> list[str]:
        statements = []
        current = []
        for line in sql_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            current.append(line)
            if stripped.endswith(";"):
                statement = "\n".join(current).rstrip(";").strip()
                if statement:
                    statements.append(statement)
                current = []
        if current:
            statements.append("\n".join(current).strip())
        return statements
