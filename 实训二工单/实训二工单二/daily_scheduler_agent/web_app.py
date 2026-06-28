# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: Flask Web 服务，提供日程提醒智能体聊天网页界面和网页提醒轮询
"""

import threading
import webbrowser
from datetime import date, datetime
from typing import Tuple

from flask import Flask, jsonify, render_template, render_template_string, request

from agent.scheduler_agent import SchedulerAgent
from config import Config
from db.execution_log_dao import ExecutionLogDAO
from db.mysql_connector import MySQLConnector
from db.schedule_dao import ScheduleDAO
from reminder.message_templates import build_reminder_message
from reminder.notification import notify_reminder
from utils.logger import get_logger
from utils.time_utils import format_schedule_time


HOST = "0.0.0.0"
PORT = 5000
APP_URL = f"http://{HOST}:{PORT}"
WEB_REMINDER_LOOKBACK_SECONDS = max(3600, Config.REMINDER_LOOKBACK_SECONDS)

app = Flask(__name__)
agent = SchedulerAgent()
schedule_dao = ScheduleDAO()
execution_log_dao = ExecutionLogDAO()
logger = get_logger("WebApp")
agent_lock = threading.Lock()
reminder_lock = threading.Lock()


@app.route("/")
def index():
    logger.info("Web page opened")
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or payload.get("query") or "").strip()
    if not message:
        logger.warning("Empty chat message received")
        return jsonify({"ok": False, "reply": "请输入要发送的内容。"}), 400

    try:
        with agent_lock:
            reply = agent.process(message)
        logger.info("Chat processed: input=%s reply=%s", message, reply)
        return jsonify({"ok": True, "reply": reply})
    except Exception as exc:
        logger.exception("Chat failed: input=%s error=%s", message, exc)
        return jsonify({"ok": False, "reply": f"处理失败：{exc}"}), 500


@app.route("/today", methods=["GET"])
def today():
    try:
        summary, count, schedules = build_today_summary()
        execution_log_dao.add_log(
            user_input="今日日程按钮",
            intent="query",
            action="query",
            target_schedule_id=None,
            result="success",
        )
        logger.info("Today schedules queried: count=%s", count)
        return jsonify({"ok": True, "reply": summary, "count": count, "schedules": schedules})
    except Exception as exc:
        try:
            execution_log_dao.add_log(
                user_input="今日日程按钮",
                intent="query",
                action="query",
                target_schedule_id=None,
                result="failed",
                error_message=str(exc),
            )
        except Exception:
            logger.exception("Failed to write today query failure log")
        logger.exception("Today schedules query failed: %s", exc)
        return jsonify({"ok": False, "reply": f"查询今日日程失败：{exc}"}), 500


@app.route("/skills", methods=["GET"])
def skills():
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；返回智能体技能列表。"""
    return jsonify({"ok": True, **agent.get_skills()})


@app.route("/status", methods=["GET"])
def status():
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；返回数据库连接状态和表记录数。"""
    return jsonify(get_database_status())


@app.route("/test_notification", methods=["GET"])
def test_notification():
    """测试 Windows 右下角弹窗通知"""
    from reminder.notification import show_windows_toast
    show_windows_toast("日程提醒", "这是一条测试通知\n如果你看到这条消息，弹窗功能正常！", duration=5)
    return jsonify({"ok": True, "reply": "测试通知已发送，请查看电脑右下角系统托盘区域。"})


@app.route("/logs", methods=["GET"])
def logs():
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；提供执行日志验收页面，展示 agent_execution_logs 最近记录。"""
    rows = execution_log_dao.list_recent(limit=50)
    return render_template_string(
        """
        <!doctype html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8">
            <title>Agent 执行日志</title>
            <style>
                body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; color: #1f2937; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }
                th { background: #f8fafc; }
                tr:nth-child(even) { background: #fbfdff; }
            </style>
        </head>
        <body>
            <h1>Agent 执行日志</h1>
            <table>
                <thead>
                    <tr>
                        <th>ID</th><th>输入</th><th>意图</th><th>动作</th>
                        <th>目标日程</th><th>结果</th><th>错误</th><th>执行时间</th>
                    </tr>
                </thead>
                <tbody>
                {% for item in rows %}
                    <tr>
                        <td>{{ item.id }}</td>
                        <td>{{ item.user_input or "" }}</td>
                        <td>{{ item.intent or "" }}</td>
                        <td>{{ item.action or "" }}</td>
                        <td>{{ item.target_schedule_id or "" }}</td>
                        <td>{{ item.result }}</td>
                        <td>{{ item.error_message or "" }}</td>
                        <td>{{ item.execution_time }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </body>
        </html>
        """,
        rows=rows,
    )


@app.route("/db_status", methods=["GET"])
def db_status():
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；返回数据库连接状态和核心表记录数。"""
    status_data = get_database_status()
    http_status = 200 if status_data["ok"] else 500
    return jsonify(status_data), http_status


def get_database_status() -> dict:
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；统计核心数据表记录数。"""
    table_names = ("schedules", "agent_execution_logs", "reminder_logs")
    try:
        counts = {}
        with MySQLConnector().get_connection() as connection:
            with connection.cursor() as cursor:
                for table_name in table_names:
                    cursor.execute(f"SELECT COUNT(*) AS total FROM {table_name}")
                    counts[table_name] = cursor.fetchone()["total"]
        return {
            "ok": True,
            "database": Config.MYSQL_DATABASE,
            "connection": "connected",
            "schedules_count": counts["schedules"],
            "agent_execution_logs_count": counts["agent_execution_logs"],
            "reminder_logs_count": counts["reminder_logs"],
            "tables": counts,
        }
    except Exception as exc:
        logger.exception("Database status check failed: %s", exc)
        return {
            "ok": False,
            "database": Config.MYSQL_DATABASE,
            "connection": "failed",
            "schedules_count": 0,
            "agent_execution_logs_count": 0,
            "reminder_logs_count": 0,
            "tables": {},
            "error": str(exc),
        }


@app.route("/reminders", methods=["GET"])
def reminders():
    """网页轮询接口：查找到点且未提醒过的日程，返回给浏览器显示。"""
    with reminder_lock:
        try:
            now = datetime.now().replace(microsecond=0)
            due_schedules = schedule_dao.get_due_schedules(now, WEB_REMINDER_LOOKBACK_SECONDS)
            reminders_data = []

            for schedule in due_schedules:
                message = build_reminder_message(schedule["content"])
                schedule_dao.add_reminder_log(schedule["id"], schedule["occurrence_time"], message)
                execution_log_dao.add_log(
                    user_input=None,
                    intent="remind",
                    action="remind",
                    target_schedule_id=schedule["id"],
                    result="success",
                )
                reminders_data.append(
                    {
                        "schedule_id": schedule["id"],
                        "message": message,
                        "scheduled_time": format_schedule_time(schedule["occurrence_time"]),
                    }
                )
                # 发送 Windows 右下角 Toast 弹窗通知
                notify_reminder(
                    schedule_id=schedule["id"],
                    content=schedule["content"],
                    scheduled_time=format_schedule_time(schedule["occurrence_time"]),
                )
                logger.info("Web reminder sent: schedule_id=%s message=%s", schedule["id"], message)

            return jsonify({"ok": True, "reminders": reminders_data})
        except Exception as exc:
            logger.exception("Web reminder polling failed: %s", exc)
            try:
                execution_log_dao.add_log(
                    user_input=None,
                    intent="remind",
                    action="remind",
                    target_schedule_id=None,
                    result="failed",
                    error_message=str(exc),
                )
            except Exception:
                logger.exception("Failed to write reminder polling failure log")
            return jsonify({"ok": False, "reply": f"提醒轮询失败：{exc}", "reminders": []}), 500


def build_today_summary() -> Tuple[str, int, list]:
    schedules = schedule_dao.get_today_schedules(date.today())
    if not schedules:
        return "你今天没有待提醒的日程。", 0, []

    lines = [
        f"{index}. #{item['id']} {format_schedule_time(item['scheduled_time'])} {item['content']}"
        for index, item in enumerate(schedules, start=1)
    ]
    items = [
        {
            "id": item["id"],
            "time": format_schedule_time(item["scheduled_time"]),
            "content": item["content"],
            "repeat_rule": item.get("repeat_rule"),
        }
        for item in schedules
    ]
    return "你今天的日程包括：\n" + "\n".join(lines), len(schedules), items


def open_browser() -> None:
    logger.info("Opening browser: %s", APP_URL)
    webbrowser.open(APP_URL)


def main() -> None:
    MySQLConnector().init_database()
    logger.info("Web app starting at %s", APP_URL)
    threading.Timer(1.0, open_browser).start()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
