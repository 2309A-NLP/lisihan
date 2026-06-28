# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 自然语言意图识别，解析增删改查和确认取消指令
"""

import re
from datetime import datetime
from typing import Dict, Optional

from utils.time_utils import (
    find_date_context_text,
    find_datetime_text,
    normalize_repeat_rule,
    parse_natural_datetime,
    parse_repeat_end_date,
)


class IntentParser:
    """面向日程提醒的轻量中文 NLU 解析器。"""

    ADD_KEYWORDS = ("添加", "新增", "新建", "创建", "提醒我", "帮我记", "安排", "设个提醒", "加一个", "记得提醒我", "记得")
    # 工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；扩展查询关键词，覆盖“我现在有什么日程”等验收说法。
    QUERY_KEYWORDS = (
        "今天的日程",
        "现在有什么日程",
        "我的日程",
        "今天的安排",
        "今天要做什么",
        "有哪些日程",
        "查看日程",
        "查询日程",
        "日程有哪些",
        "安排有哪些",
        "提醒列表",
    )
    DELETE_KEYWORDS = ("删除", "取消", "移除", "删掉", "去掉")
    UPDATE_KEYWORDS = ("修改", "更改", "改成", "调整", "更新")

    def parse(self, text: str, now: Optional[datetime] = None) -> Dict:
        text = text.strip()
        now = now or datetime.now()
        if not text:
            return {"intent": "empty", "raw_text": text}

        if self._is_confirm(text):
            return {"intent": "confirm", "raw_text": text}
        if self._is_cancel_confirm(text):
            return {"intent": "cancel_confirm", "raw_text": text}
        if self._match_any(text, self.DELETE_KEYWORDS):
            return self._parse_delete(text)
        if self._match_any(text, self.UPDATE_KEYWORDS):
            return self._parse_update(text, now)
        if self._match_any(text, self.QUERY_KEYWORDS) or ("日程" in text and re.search(r"今天|明天|最近|全部|所有", text)):
            return self._parse_query(text)
        if self._match_any(text, self.ADD_KEYWORDS) or find_datetime_text(text) or find_date_context_text(text):
            return self._parse_add(text, now)

        return {"intent": "unknown", "raw_text": text}

    def _parse_add(self, text: str, now: datetime) -> Dict:
        time_text = find_datetime_text(text)
        date_context_text = None if time_text else find_date_context_text(text)
        scheduled_time = parse_natural_datetime(time_text, now) if time_text else None
        repeat_rule = normalize_repeat_rule(text)
        repeat_end_date = parse_repeat_end_date(text, now)
        content = self._extract_content(text, time_text or date_context_text)
        return {
            "intent": "add",
            "raw_text": text,
            "content": content,
            "scheduled_time": scheduled_time,
            "repeat_rule": repeat_rule,
            "repeat_end_date": repeat_end_date,
            "time_text": time_text,
            "date_context_text": date_context_text,
        }

    def _parse_query(self, text: str) -> Dict:
        scope = "today"
        if "明天" in text:
            scope = "tomorrow"
        elif "全部" in text or "所有" in text:
            scope = "all"
        elif "最近" in text or "接下来" in text or "未来" in text:
            scope = "upcoming"
        return {"intent": "query", "raw_text": text, "scope": scope}

    def _parse_delete(self, text: str) -> Dict:
        target_ids = self._extract_target_ids(text)
        target_id = target_ids[0] if target_ids else None
        target_content = self._clean_delete_text(text, target_ids)
        return {
            "intent": "delete",
            "raw_text": text,
            "target_id": target_id,
            "target_ids": target_ids,
            "target_content": target_content,
        }

    def _parse_update(self, text: str, now: datetime) -> Dict:
        target_id = self._extract_target_id(text)
        time_text = find_datetime_text(text)
        scheduled_time = parse_natural_datetime(time_text, now) if time_text else None
        content = self._extract_content(text, time_text)
        content = re.sub(r"^(修改|更改|调整|更新)?\s*(日程|提醒)?\s*#?\d+\s*(为|成|改成|到)?", "", content or "").strip()
        return {
            "intent": "update",
            "raw_text": text,
            "target_id": target_id,
            "content": content or None,
            "scheduled_time": scheduled_time,
            "repeat_rule": normalize_repeat_rule(text),
            "repeat_end_date": parse_repeat_end_date(text, now),
            "time_text": time_text,
        }

    def _extract_content(self, text: str, time_text: Optional[str]) -> Optional[str]:
        cleaned = text
        for keyword in sorted(self.ADD_KEYWORDS + self.UPDATE_KEYWORDS, key=len, reverse=True):
            cleaned = cleaned.replace(keyword, "")
        if time_text:
            cleaned = cleaned.replace(time_text, "")
        cleaned = re.sub(r"(日程|提醒|事项|：|:)", "", cleaned)
        cleaned = re.sub(r"(每天|每日|天天|每周|每星期|每礼拜|每月|daily|weekly|monthly)", "", cleaned, flags=re.I)
        cleaned = re.sub(r"(到|截止到|截至|直到)\d{4}?[年/-]?\d{1,2}[月/-]\d{1,2}[日号]?", "", cleaned)
        cleaned = re.sub(r"^(我|请|麻烦|帮忙|帮我)+", "", cleaned)
        cleaned = re.sub(r"(，|,|。|\.)?(我|请|麻烦|帮忙|帮我)?$", "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = cleaned.strip("，。,.；; ")
        return cleaned or None

    def _extract_target_id(self, text: str) -> Optional[int]:
        match = re.search(r"(?:日程|编号|#)\s*(\d+)", text)
        return int(match.group(1)) if match else None

    def _extract_target_ids(self, text: str) -> list[int]:
        """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；支持“删除日程3,4 / 3和4 / 3 4”的批量编号。"""
        ids = [int(value) for value in re.findall(r"#\s*(\d+)", text)]
        match = re.search(r"(?:日程|编号)\s*([#\d,，、和\s]+)", text)
        if match:
            ids.extend(int(value) for value in re.findall(r"\d+", match.group(1)))
        unique_ids = []
        for schedule_id in ids:
            if schedule_id not in unique_ids:
                unique_ids.append(schedule_id)
        return unique_ids

    def _clean_delete_text(self, text: str, target_ids: list[int]) -> Optional[str]:
        cleaned = text
        for keyword in self.DELETE_KEYWORDS:
            cleaned = cleaned.replace(keyword, "")
        cleaned = cleaned.replace("日程", "").replace("提醒", "")
        for target_id in target_ids:
            cleaned = re.sub(rf"#?\s*{target_id}", "", cleaned)
        cleaned = re.sub(r"[,，、和]+", "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned).strip("，。,.；; ")
        return cleaned or None

    def _match_any(self, text: str, keywords) -> bool:
        return any(keyword in text for keyword in keywords)

    def _is_confirm(self, text: str) -> bool:
        return text in {"确认", "确认删除", "是", "好的", "好", "删除吧", "确定", "yes", "y"}

    def _is_cancel_confirm(self, text: str) -> bool:
        return text in {"取消", "不删", "别删", "算了", "否", "不要", "取消删除", "no", "n"}
