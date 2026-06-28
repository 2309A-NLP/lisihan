# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 后台守护进程入口，常驻运行并轮询数据库发送提醒
"""

import signal
import threading

from db.execution_log_dao import ExecutionLogDAO
from db.mysql_connector import MySQLConnector
from reminder.reminder_service import ReminderService
from utils.logger import get_logger


stop_event = threading.Event()


def _handle_stop(signum, frame) -> None:
    stop_event.set()


def main() -> None:
    logger = get_logger("SchedulerDaemon")
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    connector = MySQLConnector()
    connector.init_database()

    log_dao = ExecutionLogDAO(connector)
    service = ReminderService(log_dao=log_dao)

    try:
        log_dao.add_log(None, "system", "daemon_start", None, "success")
        logger.info("Scheduler daemon starting")
        service.run_once()
        service.start()
        stop_event.wait()
    except Exception as exc:
        logger.exception("Scheduler daemon failed: %s", exc)
        try:
            log_dao.add_log(None, "system", "daemon_run", None, "failed", str(exc))
        except Exception:
            logger.exception("Failed to write daemon failure log")
        raise
    finally:
        service.stop()
        try:
            log_dao.add_log(None, "system", "daemon_stop", None, "success")
        except Exception:
            logger.exception("Failed to write daemon stop log")
        logger.info("Scheduler daemon stopped")


if __name__ == "__main__":
    main()
