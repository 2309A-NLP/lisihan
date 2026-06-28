# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 交互式命令行入口，处理用户自然语言日程增删改查
"""

from agent.scheduler_agent import SchedulerAgent
from db.mysql_connector import MySQLConnector
from utils.logger import get_logger


def main() -> None:
    logger = get_logger("Interactive")
    MySQLConnector().init_database()
    agent = SchedulerAgent()

    print("欢迎使用日程提醒智能体！")
    print("输入 exit、quit 或 q 退出交互模式。")
    try:
        while True:
            user_input = input("> ").strip()
            if user_input.lower() in {"exit", "quit", "q", "退出"}:
                print("智能体：再见。后台提醒进程不受影响。")
                break
            response = agent.process(user_input)
            print(f"智能体：{response}")
    except KeyboardInterrupt:
        print("\n智能体：已退出交互模式。后台提醒进程不受影响。")
    except Exception as exc:
        logger.exception("Interactive mode failed: %s", exc)
        print(f"智能体：执行失败，错误信息：{exc}")


if __name__ == "__main__":
    main()
