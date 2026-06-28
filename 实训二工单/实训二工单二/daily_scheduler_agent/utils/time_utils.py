# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 中文自然语言时间解析和循环日程发生时间计算
"""

import calendar
import re
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple


CN_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

WEEKDAY_MAP = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
}

NUM_PATTERN = r"(?:\d{1,2}|[零一二两三四五六七八九十]{1,3})"
TIME_PATTERN = rf"(?:早上|上午|中午|下午|晚上|傍晚|凌晨)?\s*{NUM_PATTERN}(?:(?:\s*[:：]\s*{NUM_PATTERN})|(?:\s*[点时](?:\s*{NUM_PATTERN})?))?\s*(?:分)?(?:半)?"
DATE_CONTEXT_PATTERN = r"(?:今天|明天|后天|大后天|(?:\d{4}[年/-])?\d{1,2}[月/-]\d{1,2}[日号]?|(?:本周|这周|下周|周|星期|礼拜|每周|每星期|每礼拜)\s*[一二三四五六日天1-7])"
PERIOD_PATTERN = r"(?:早上|上午|中午|下午|晚上|傍晚|凌晨)"


def cn_to_int(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if value in CN_NUMBERS:
        return CN_NUMBERS[value]
    if value == "十一":
        return 11
    if value == "十二":
        return 12
    if value.startswith("十"):
        return 10 + CN_NUMBERS.get(value[1:], 0)
    if "十" in value:
        left, _, right = value.partition("十")
        return CN_NUMBERS.get(left, 1) * 10 + CN_NUMBERS.get(right, 0)
    raise ValueError(f"无法解析数字: {value}")


def normalize_repeat_rule(text: str) -> Optional[str]:
    lowered = text.lower()
    if re.search(r"每天|每日|天天|daily", lowered):
        return "daily"
    if re.search(r"每周|每星期|每礼拜|weekly", lowered):
        return "weekly"
    if re.search(r"每月|monthly", lowered):
        return "monthly"
    return None


def parse_repeat_end_date(text: str, now: Optional[datetime] = None) -> Optional[date]:
    now = now or datetime.now()
    match = re.search(r"(?:到|截止到|截至|直到)(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?", text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = re.search(r"(?:到|截止到|截至|直到)(\d{1,2})[月/-](\d{1,2})[日号]?", text)
    if match:
        result = date(now.year, int(match.group(1)), int(match.group(2)))
        if result < now.date():
            result = date(now.year + 1, result.month, result.day)
        return result
    return None


def find_datetime_text(text: str) -> Optional[str]:
    patterns = [
        rf"(?:\d{{4}}[年/-])?\d{{1,2}}[月/-]\d{{1,2}}[日号]?\s*{TIME_PATTERN}",
        rf"(?:今天|明天|后天|大后天)?\s*{TIME_PATTERN}",
        rf"(?:本周|这周|下周|周|星期|礼拜|每周|每星期|每礼拜)\s*[一二三四五六日天1-7]\s*{TIME_PATTERN}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def find_date_context_text(text: str) -> Optional[str]:
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；记录只有日期/时段但缺少具体时刻的上下文。"""
    patterns = [
        rf"{DATE_CONTEXT_PATTERN}\s*{PERIOD_PATTERN}?",
        PERIOD_PATTERN,
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    return None


def _parse_date_part(text: str, now: datetime) -> date:
    if "大后天" in text:
        return now.date() + timedelta(days=3)
    if "后天" in text:
        return now.date() + timedelta(days=2)
    if "明天" in text:
        return now.date() + timedelta(days=1)
    if "今天" in text:
        return now.date()

    match = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?", text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    match = re.search(r"(\d{1,2})[月/-](\d{1,2})[日号]?", text)
    if match:
        result = date(now.year, int(match.group(1)), int(match.group(2)))
        if result < now.date():
            result = date(now.year + 1, result.month, result.day)
        return result

    match = re.search(r"(本周|这周|下周|周|星期|礼拜|每周|每星期|每礼拜)\s*([一二三四五六日天1-7])", text)
    if match:
        target = WEEKDAY_MAP[match.group(2)]
        current = now.weekday()
        if match.group(1) == "下周":
            return now.date() + timedelta(days=(7 - current + target))
        delta = (target - current) % 7
        return now.date() + timedelta(days=delta)

    return now.date()


def _parse_time_part(text: str) -> Optional[time]:
    pattern = rf"(早上|上午|中午|下午|晚上|傍晚|凌晨)?\s*({NUM_PATTERN})(?:(?:\s*[:：]\s*({NUM_PATTERN}))|(?:\s*[点时]\s*({NUM_PATTERN})?))?\s*(分)?\s*(半)?"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    match = matches[-1]
    period = match.group(1) or ""
    hour = cn_to_int(match.group(2))
    minute_text = match.group(3) or match.group(4)
    minute = cn_to_int(minute_text) if minute_text else 0
    if match.group(6):
        minute = 30

    if period in {"下午", "晚上", "傍晚"} and hour < 12:
        hour += 12
    if period == "中午" and hour < 11:
        hour += 12
    if period == "凌晨" and hour == 12:
        hour = 0

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return time(hour=hour, minute=minute)
    return None


def parse_natural_datetime(text: Optional[str], now: Optional[datetime] = None) -> Optional[datetime]:
    if not text:
        return None
    now = now or datetime.now()
    parsed_time = _parse_time_part(text)
    if not parsed_time:
        return None
    parsed_date = _parse_date_part(text, now)
    result = datetime.combine(parsed_date, parsed_time)
    if result < now and not re.search(r"今天|明天|后天|大后天|\d{1,2}[月/-]\d{1,2}|周|星期|礼拜", text):
        result += timedelta(days=1)
    return result


def next_month_same_day(value: datetime) -> datetime:
    month = value.month + 1
    year = value.year
    if month == 13:
        month = 1
        year += 1
    max_day = calendar.monthrange(year, month)[1]
    return value.replace(year=year, month=month, day=min(value.day, max_day))


def calculate_occurrence(schedule_time: datetime, repeat_rule: Optional[str], target: datetime) -> Optional[datetime]:
    if not repeat_rule:
        return schedule_time
    if target.date() < schedule_time.date():
        return None

    occurrence = datetime.combine(target.date(), schedule_time.time())
    if repeat_rule == "daily":
        return occurrence
    if repeat_rule == "weekly" and target.weekday() == schedule_time.weekday():
        return occurrence
    if repeat_rule == "monthly" and target.day == schedule_time.day:
        return occurrence
    return None


def format_schedule_time(value: datetime, today: Optional[date] = None) -> str:
    today = today or datetime.now().date()
    if value.date() == today:
        prefix = "今天"
    elif value.date() == today + timedelta(days=1):
        prefix = "明天"
    elif value.date() == today + timedelta(days=2):
        prefix = "后天"
    else:
        prefix = value.strftime("%Y-%m-%d")
    return f"{prefix} {value.strftime('%H:%M')}"


def _format_parse_time(value: datetime, today: Optional[date] = None) -> str:
    today = today or datetime.now().date()
    if value.date() == today:
        prefix = "今天"
    elif value.date() == today + timedelta(days=1):
        prefix = "明天"
    elif value.date() == today + timedelta(days=2):
        prefix = "后天"
    elif value.date() == today + timedelta(days=3):
        prefix = "大后天"
    else:
        prefix = value.strftime("%Y-%m-%d")
    return f"{prefix}{value.strftime('%H:%M')}"


def parse_time(text: str, now: Optional[datetime] = None) -> dict:
    """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；同时抽取自然语言中的提醒时间和剩余事项内容。"""
    now = now or datetime.now()
    time_text = find_datetime_text(text)
    scheduled_time = parse_natural_datetime(time_text, now=now) if time_text else None
    content = text
    if time_text:
        content = content.replace(time_text, "", 1)
    content = re.sub(r"^(日程|提醒|事项|：|:)", "", content)
    content = re.sub(r"\s+", "", content).strip("，。,.；; ")
    return {
        "time": _format_parse_time(scheduled_time, today=now.date()) if scheduled_time else None,
        "content": content or None,
    }


def split_datetime_from_text(text: str, now: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[str]]:
    time_text = find_datetime_text(text)
    if not time_text:
        return None, None
    return parse_natural_datetime(time_text, now=now), time_text
