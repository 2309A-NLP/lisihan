# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: Agent 执行日志 DAO，记录用户操作、后台提醒和异常信息
"""

from typing import Dict, List, Optional

from db.mysql_connector import MySQLConnector


class ExecutionLogDAO:
    """写入和查询 agent_execution_logs。"""

    def __init__(self, connector: Optional[MySQLConnector] = None):
        self.connector = connector or MySQLConnector()

    def add_log(
        self,
        user_input: Optional[str],
        intent: Optional[str],
        action: Optional[str],
        target_schedule_id: Optional[int],
        result: str,
        error_message: Optional[str] = None,
    ) -> int:
        sql = """
            INSERT INTO agent_execution_logs
                (user_input, intent, action, target_schedule_id, result, error_message)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.connector.get_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        sql,
                        (
                            user_input,
                            intent,
                            action,
                            target_schedule_id,
                            result,
                            error_message,
                        ),
                    )
                    log_id = cursor.lastrowid
                connection.commit()
                return log_id
            except Exception:
                connection.rollback()
                raise

    def list_recent(self, limit: int = 20) -> List[Dict]:
        sql = """
            SELECT id, user_input, intent, action, target_schedule_id, result,
                   error_message, execution_time
            FROM agent_execution_logs
            ORDER BY execution_time DESC
            LIMIT %s
        """
        with self.connector.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (limit,))
                return list(cursor.fetchall())
