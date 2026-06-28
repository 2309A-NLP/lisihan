# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 日程 CRUD、到期任务查询和提醒日志 DAO
"""

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

from db.mysql_connector import MySQLConnector
from utils.time_utils import calculate_occurrence, next_month_same_day


class ScheduleDAO:
    """通过 MySQL 完成日程和提醒记录的数据访问。"""

    def __init__(self, connector: Optional[MySQLConnector] = None):
        self.connector = connector or MySQLConnector()

    def add_schedule(
        self,
        content: str,
        scheduled_time: datetime,
        repeat_rule: Optional[str] = None,
        repeat_end_date: Optional[date] = None,
    ) -> int:
        sql = """
            INSERT INTO schedules (content, scheduled_time, repeat_rule, repeat_end_date, status)
            VALUES (%s, %s, %s, %s, 1)
        """
        with self.connector.get_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, (content, scheduled_time, repeat_rule, repeat_end_date))
                    schedule_id = cursor.lastrowid
                connection.commit()
                return schedule_id
            except Exception:
                connection.rollback()
                raise

    def get_today_schedules(self, target_date: date) -> List[Dict]:
        rows = self.get_all_active_schedules()
        target_dt = datetime.combine(target_date, time.min)
        result = []
        for row in rows:
            occurrence = calculate_occurrence(row["scheduled_time"], row.get("repeat_rule"), target_dt)
            if not occurrence:
                continue
            if row.get("repeat_end_date") and occurrence.date() > row["repeat_end_date"]:
                continue
            if not row.get("repeat_rule") and occurrence.date() != target_date:
                continue
            item = dict(row)
            item["occurrence_time"] = occurrence
            item["scheduled_time"] = occurrence
            result.append(item)
        return sorted(result, key=lambda item: item["scheduled_time"])

    def get_all_active_schedules(self) -> List[Dict]:
        sql = """
            SELECT id, content, scheduled_time, repeat_rule, repeat_end_date, status, created_at, updated_at
            FROM schedules
            WHERE status = 1
            ORDER BY scheduled_time ASC
        """
        with self.connector.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                return list(cursor.fetchall())

    def get_schedule_by_id(self, schedule_id: int) -> Optional[Dict]:
        sql = """
            SELECT id, content, scheduled_time, repeat_rule, repeat_end_date, status, created_at, updated_at
            FROM schedules
            WHERE id = %s AND status = 1
        """
        with self.connector.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (schedule_id,))
                return cursor.fetchone()

    def find_active_by_content(self, keyword: str, limit: int = 5) -> List[Dict]:
        sql = """
            SELECT id, content, scheduled_time, repeat_rule, repeat_end_date, status, created_at, updated_at
            FROM schedules
            WHERE status = 1 AND content LIKE %s
            ORDER BY scheduled_time ASC
            LIMIT %s
        """
        with self.connector.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (f"%{keyword}%", limit))
                return list(cursor.fetchall())

    def delete_schedule_by_id(self, schedule_id: int, confirm: bool = True) -> Optional[Dict]:
        if not confirm:
            return None
        schedule = self.get_schedule_by_id(schedule_id)
        if not schedule:
            return None
        sql = "UPDATE schedules SET status = 0 WHERE id = %s AND status = 1"
        with self.connector.get_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, (schedule_id,))
                connection.commit()
                return schedule
            except Exception:
                connection.rollback()
                raise

    def update_schedule(self, schedule_id: int, **kwargs) -> Optional[Dict]:
        allowed = {"content", "scheduled_time", "repeat_rule", "repeat_end_date", "status"}
        fields = {key: value for key, value in kwargs.items() if key in allowed and value is not None}
        if not fields:
            return self.get_schedule_by_id(schedule_id)

        assignments = ", ".join(f"{key} = %s" for key in fields)
        values = list(fields.values()) + [schedule_id]
        sql = f"UPDATE schedules SET {assignments} WHERE id = %s AND status = 1"
        with self.connector.get_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, values)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self.get_schedule_by_id(schedule_id)

    def get_upcoming_schedules(self, limit: int = 10) -> List[Dict]:
        now = datetime.now()
        result = []
        for row in self.get_all_active_schedules():
            occurrence = self._next_occurrence(row, now)
            if occurrence:
                item = dict(row)
                item["occurrence_time"] = occurrence
                item["scheduled_time"] = occurrence
                result.append(item)
        return sorted(result, key=lambda item: item["scheduled_time"])[:limit]

    def get_due_schedules(self, now: datetime, lookback_seconds: int = 60) -> List[Dict]:
        window_start = now - timedelta(seconds=lookback_seconds)
        due = []
        for row in self.get_all_active_schedules():
            occurrence = calculate_occurrence(row["scheduled_time"], row.get("repeat_rule"), now)
            if not occurrence:
                continue
            if row.get("repeat_end_date") and occurrence.date() > row["repeat_end_date"]:
                continue
            if window_start <= occurrence <= now and not self.has_reminder_log(row["id"], occurrence):
                item = dict(row)
                item["occurrence_time"] = occurrence
                item["scheduled_time"] = occurrence
                due.append(item)
        return due

    def has_reminder_log(self, schedule_id: int, occurrence_time: datetime) -> bool:
        sql = "SELECT id FROM reminder_logs WHERE schedule_id = %s AND occurrence_time = %s LIMIT 1"
        with self.connector.get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (schedule_id, occurrence_time))
                return cursor.fetchone() is not None

    def add_reminder_log(self, schedule_id: int, occurrence_time: datetime, reminder_content: str) -> None:
        sql = """
            INSERT IGNORE INTO reminder_logs (schedule_id, occurrence_time, reminder_content)
            VALUES (%s, %s, %s)
        """
        with self.connector.get_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, (schedule_id, occurrence_time, reminder_content))
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _next_occurrence(self, row: Dict, now: datetime) -> Optional[datetime]:
        base_time = row["scheduled_time"]
        repeat_rule = row.get("repeat_rule")
        repeat_end = row.get("repeat_end_date")
        if not repeat_rule:
            return base_time if base_time >= now else None

        candidate = base_time
        while candidate < now:
            if repeat_rule == "daily":
                candidate += timedelta(days=1)
            elif repeat_rule == "weekly":
                candidate += timedelta(days=7)
            elif repeat_rule == "monthly":
                candidate = next_month_same_day(candidate)
            else:
                return None
        if repeat_end and candidate.date() > repeat_end:
            return None
        return candidate
