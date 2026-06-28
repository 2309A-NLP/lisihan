# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 项目配置，集中管理数据库、轮询、日志和进程文件路径
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Config:
    """日程提醒智能体的集中配置。"""

    MYSQL_HOST = "127.0.0.1"
    MYSQL_PORT = 3306
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "root"
    MYSQL_DATABASE = "schedule_db"
    MYSQL_CHARSET = "utf8mb4"

    REMINDER_POLL_SECONDS = 30
    REMINDER_LOOKBACK_SECONDS = 60

    LOG_DIR = BASE_DIR / "logs"
    LOG_FILE = LOG_DIR / "agent.log"
    PID_FILE = LOG_DIR / "scheduler_agent.pid"
