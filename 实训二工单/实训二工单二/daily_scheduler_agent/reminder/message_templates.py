# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 提醒消息模板
"""

import random


TEMPLATES = [
    "{content} 的时间到啦。",
    "现在该处理：{content}。",
    "提醒：{content}。",
]


def build_reminder_message(content: str) -> str:
    return random.choice(TEMPLATES).format(content=content)
