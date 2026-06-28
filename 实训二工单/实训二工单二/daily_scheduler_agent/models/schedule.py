# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 日程数据模型定义
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Optional


@dataclass
class Schedule:
    """表示一条日程记录。"""

    id: Optional[int]
    content: str
    scheduled_time: datetime
    repeat_rule: Optional[str] = None
    repeat_end_date: Optional[date] = None
    status: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Schedule":
        return cls(
            id=row.get("id"),
            content=row["content"],
            scheduled_time=row["scheduled_time"],
            repeat_rule=row.get("repeat_rule"),
            repeat_end_date=row.get("repeat_end_date"),
            status=row.get("status", 1),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "scheduled_time": self.scheduled_time,
            "repeat_rule": self.repeat_rule,
            "repeat_end_date": self.repeat_end_date,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
