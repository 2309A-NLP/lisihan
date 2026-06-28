# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: APScheduler 后台轮询服务，检查到期日程并发送提醒
"""

from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import Config
from db.execution_log_dao import ExecutionLogDAO
from db.schedule_dao import ScheduleDAO
from reminder.message_templates import build_reminder_message
from utils.logger import get_logger


class ReminderService:
    """每 30 秒轮询数据库，触发到点提醒并写入提醒日志。"""

    def __init__(
        self,
        dao: Optional[ScheduleDAO] = None,
        log_dao: Optional[ExecutionLogDAO] = None,
        callback: Optional[Callable[[str], None]] = None,
        poll_seconds: int = Config.REMINDER_POLL_SECONDS,
        lookback_seconds: int = Config.REMINDER_LOOKBACK_SECONDS,
    ):
        self.dao = dao or ScheduleDAO()
        self.log_dao = log_dao or ExecutionLogDAO()
        self.callback = callback or self._default_callback
        self.poll_seconds = poll_seconds
        self.lookback_seconds = lookback_seconds
        self.logger = get_logger(self.__class__.__name__)
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        if self.scheduler.running:
            return
        self.scheduler.add_job(
            self.check_and_remind,
            trigger=IntervalTrigger(seconds=self.poll_seconds),
            id="schedule_reminder_polling",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()
        self.logger.info("Reminder service started, polling every %s seconds", self.poll_seconds)

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            self.logger.info("Reminder service stopped")

    def check_and_remind(self) -> None:
        now = datetime.now().replace(microsecond=0)
        try:
            schedules = self.dao.get_due_schedules(now, self.lookback_seconds)
            for schedule in schedules:
                message = build_reminder_message(schedule["content"])
                self.callback(message)
                self.dao.add_reminder_log(schedule["id"], schedule["occurrence_time"], message)
                self.log_dao.add_log(
                    user_input=None,
                    intent="remind",
                    action="remind",
                    target_schedule_id=schedule["id"],
                    result="success",
                )
                self.logger.info("Reminder sent for schedule #%s: %s", schedule["id"], message)
        except Exception as exc:
            self.logger.exception("Reminder polling failed: %s", exc)
            try:
                self.log_dao.add_log(
                    user_input=None,
                    intent="remind",
                    action="remind",
                    target_schedule_id=None,
                    result="failed",
                    error_message=str(exc),
                )
            except Exception:
                self.logger.exception("Failed to write reminder failure log")

    def run_once(self) -> None:
        self.check_and_remind()

    def _default_callback(self, message: str) -> None:
        self.logger.info("Reminder: %s", message)
