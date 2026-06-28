# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 日程提醒智能体主流程，处理自然语言增删改查并记录执行日志
"""

from datetime import date, timedelta
from typing import Dict, List, Optional

from agent.intent_parser import IntentParser
from db.execution_log_dao import ExecutionLogDAO
from db.schedule_dao import ScheduleDAO
from utils.time_utils import find_date_context_text, format_schedule_time
from utils.validator import InputValidator


class SchedulerAgent:
    """调用解析器、校验器和 DAO 完成自然语言日程管理。"""

    def __init__(
        self,
        dao: Optional[ScheduleDAO] = None,
        parser: Optional[IntentParser] = None,
        validator: Optional[InputValidator] = None,
        log_dao: Optional[ExecutionLogDAO] = None,
    ):
        self.dao = dao or ScheduleDAO()
        self.parser = parser or IntentParser()
        self.validator = validator or InputValidator()
        self.log_dao = log_dao or ExecutionLogDAO()
        self.pending_add: Optional[Dict] = None
        self.pending_delete: Optional[Dict] = None
        self.last_query_results: List[Dict] = []
        self._last_target_schedule_id: Optional[int] = None
        self._last_date_context_text: Optional[str] = None

    def get_skills(self) -> Dict:
        """工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；返回技能列表供 Web 接口展示。"""
        return {
            "skills": [
                {
                    "name": "添加日程",
                    "triggers": ["添加", "提醒我", "帮我记", "记住"],
                    "examples": ["提醒我明天下午5点开会", "添加今天晚上8点看书"],
                },
                {
                    "name": "查询日程",
                    "triggers": ["查看", "有什么日程", "今天的安排", "现在有什么"],
                    "examples": ["我今天有什么日程", "现在有什么安排"],
                },
                {
                    "name": "删除日程",
                    "triggers": ["删除", "取消", "移除"],
                    "examples": ["删除日程1", "取消下午5点开会"],
                },
                {
                    "name": "修改日程",
                    "triggers": ["修改", "改", "调整"],
                    "examples": ["修改日程1为明天下午3点开会", "把日程2调整到晚上8点"],
                },
                {
                    "name": "循环日程",
                    "triggers": ["每天", "每周", "每月", "工作日"],
                    "examples": ["每天上午8点起床", "每周一上午9点开例会"],
                },
            ]
        }

    def process(self, user_input: str) -> str:
        pending_action = "delete" if self.pending_delete else "add" if self.pending_add else None
        parsed = self.parser.parse(user_input)
        action = self._resolve_action(parsed.get("intent"), pending_action)
        self._last_target_schedule_id = None

        try:
            response = self._dispatch(user_input, parsed)
            self.log_dao.add_log(
                user_input=user_input,
                intent=parsed.get("intent"),
                action=action,
                target_schedule_id=self._last_target_schedule_id,
                result="success",
            )
            return response
        except Exception as exc:
            self.log_dao.add_log(
                user_input=user_input,
                intent=parsed.get("intent"),
                action=action,
                target_schedule_id=self._last_target_schedule_id,
                result="failed",
                error_message=str(exc),
            )
            raise

    def _dispatch(self, user_input: str, parsed: Dict) -> str:
        if self.pending_delete:
            return self._handle_pending_delete(parsed)
        if self.pending_add:
            return self._continue_pending_add(user_input)

        intent = parsed.get("intent")
        if intent == "add":
            return self._handle_add(parsed)
        if intent == "query":
            return self._handle_query(parsed)
        if intent == "delete":
            return self._handle_delete(parsed)
        if intent == "update":
            return self._handle_update(parsed)
        if intent == "empty":
            return "请输入日程指令，例如：添加日程：下午5点开会。"
        return "我还没理解你的意思。你可以说：添加日程、查看今天的日程、删除日程 #3。"

    def _handle_add(self, parsed: Dict) -> str:
        missing = self.validator.validate_add_payload(parsed)
        if missing:
            if parsed.get("date_context_text"):
                # 工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务；记住“明天早上”这类缺具体时刻的日期上下文。
                self._last_date_context_text = parsed["date_context_text"]
            self.pending_add = parsed
            return self.validator.build_add_prompt(missing)

        self._last_date_context_text = None
        schedule_id = self.dao.add_schedule(
            content=parsed["content"],
            scheduled_time=parsed["scheduled_time"],
            repeat_rule=parsed.get("repeat_rule"),
            repeat_end_date=parsed.get("repeat_end_date"),
        )
        self._last_target_schedule_id = schedule_id
        return self._build_add_success(schedule_id, parsed)

    def _continue_pending_add(self, user_input: str) -> str:
        parsed = self.parser.parse(user_input)
        pending = self.pending_add or {"intent": "add"}
        date_context_text = pending.get("date_context_text") or self._last_date_context_text
        has_new_date_context = find_date_context_text(user_input) is not None

        if parsed.get("scheduled_time"):
            if date_context_text and not has_new_date_context:
                combined = self.parser.parse(f"{date_context_text}{user_input}")
                pending["scheduled_time"] = combined.get("scheduled_time") or parsed["scheduled_time"]
                pending["time_text"] = combined.get("time_text") or parsed.get("time_text")
            else:
                pending["scheduled_time"] = parsed["scheduled_time"]
                pending["time_text"] = parsed.get("time_text")
        elif not pending.get("scheduled_time"):
            time_parsed = self.parser.parse(f"提醒我{user_input}")
            if time_parsed.get("scheduled_time"):
                pending["scheduled_time"] = time_parsed["scheduled_time"]
                pending["time_text"] = time_parsed.get("time_text")
            elif date_context_text:
                combined = self.parser.parse(f"{date_context_text}{user_input}")
                if combined.get("scheduled_time"):
                    pending["scheduled_time"] = combined["scheduled_time"]
                    pending["time_text"] = combined.get("time_text")

        if parsed.get("content"):
            pending["content"] = parsed["content"]
        elif not pending.get("content") and parsed.get("intent") in {"unknown", "empty"}:
            pending["content"] = user_input.strip()

        pending["repeat_rule"] = parsed.get("repeat_rule") or pending.get("repeat_rule")
        pending["repeat_end_date"] = parsed.get("repeat_end_date") or pending.get("repeat_end_date")
        pending["date_context_text"] = parsed.get("date_context_text") or pending.get("date_context_text")
        if pending.get("date_context_text"):
            self._last_date_context_text = pending["date_context_text"]

        missing = self.validator.validate_add_payload(pending)
        if missing:
            self.pending_add = pending
            return self.validator.build_add_prompt(missing)

        self.pending_add = None
        self._last_date_context_text = None
        schedule_id = self.dao.add_schedule(
            content=pending["content"],
            scheduled_time=pending["scheduled_time"],
            repeat_rule=pending.get("repeat_rule"),
            repeat_end_date=pending.get("repeat_end_date"),
        )
        self._last_target_schedule_id = schedule_id
        return self._build_add_success(schedule_id, pending)

    def _handle_query(self, parsed: Dict) -> str:
        scope = parsed.get("scope", "today")
        if scope == "tomorrow":
            schedules = self.dao.get_today_schedules(date.today() + timedelta(days=1))
            title = "你明天的日程包括"
        elif scope == "all":
            schedules = self.dao.get_all_active_schedules()
            title = "你的全部有效日程包括"
        elif scope == "upcoming":
            schedules = self.dao.get_upcoming_schedules()
            title = "接下来最近的日程包括"
        else:
            schedules = self.dao.get_today_schedules(date.today())
            title = "你今天的日程包括"

        self.last_query_results = schedules
        if not schedules:
            return title.replace("包括", "为空") + "。"
        return f"{title}：" + "；".join(
            f"{index}. #{item['id']} {format_schedule_time(item['scheduled_time'])} {item['content']}{self._repeat_label(item)}"
            for index, item in enumerate(schedules, start=1)
        )

    def _handle_delete(self, parsed: Dict) -> str:
        missing = self.validator.validate_delete_payload(parsed)
        if missing:
            return self.validator.build_delete_prompt()

        target_ids = parsed.get("target_ids") or []
        if len(target_ids) > 1:
            schedules, missing_ids = self._resolve_schedules_by_ids(target_ids)
            if missing_ids:
                return "没有找到这些有效日程编号：" + "、".join(f"#{item}" for item in missing_ids)
            self.pending_delete = schedules
            self._last_target_schedule_id = schedules[0]["id"]
            return "请确认是否批量删除以下日程：" + "；".join(
                self._format_item(item) for item in schedules
            ) + "。回复“确认删除”即可全部删除，回复“取消”放弃。"

        schedule = self._resolve_schedule(parsed.get("target_id"), parsed.get("target_content"))
        if isinstance(schedule, list):
            self.last_query_results = schedule
            return "我找到多条相似日程，请告诉我要删除哪一条：" + "；".join(
                f"{index}. #{item['id']} {format_schedule_time(item['scheduled_time'])} {item['content']}"
                for index, item in enumerate(schedule, start=1)
            )
        if not schedule:
            return "没有找到匹配的有效日程，请检查编号或事项内容。"

        self.pending_delete = schedule
        self._last_target_schedule_id = schedule["id"]
        return f"请确认是否删除日程：{self._format_item(schedule)}？回复“确认删除”即可删除，回复“取消”放弃。"

    def _handle_pending_delete(self, parsed: Dict) -> str:
        if parsed.get("intent") == "confirm":
            pending = self.pending_delete
            self.pending_delete = None
            schedules = pending if isinstance(pending, list) else [pending]
            deleted_items = []
            for schedule in schedules:
                deleted = self.dao.delete_schedule_by_id(schedule["id"], confirm=True)
                if deleted:
                    deleted_items.append(deleted)
            self._last_target_schedule_id = schedules[0]["id"] if schedules else None
            if not deleted_items:
                return "这些日程已经不存在或已被删除。"
            if len(deleted_items) == 1:
                return f"已删除日程，删除的内容是：{self._format_item(deleted_items[0])}"
            return "已批量删除日程：" + "；".join(self._format_item(item) for item in deleted_items)
        if parsed.get("intent") == "cancel_confirm":
            self._last_target_schedule_id = self._first_pending_delete_id()
            self.pending_delete = None
            return "已取消删除操作。"
        self._last_target_schedule_id = self._first_pending_delete_id()
        return "删除操作需要二次确认。请回复“确认删除”或“取消”。"

    def _handle_update(self, parsed: Dict) -> str:
        schedule = self._resolve_schedule(parsed.get("target_id"), None)
        if not schedule or isinstance(schedule, list):
            return "请告诉我要修改哪条日程，例如：修改日程 #3 为明天下午3点开会。"

        updates = {
            "content": parsed.get("content"),
            "scheduled_time": parsed.get("scheduled_time"),
            "repeat_rule": parsed.get("repeat_rule"),
            "repeat_end_date": parsed.get("repeat_end_date"),
        }
        updates = {key: value for key, value in updates.items() if value is not None}
        if not updates:
            return "请告诉我要修改的时间或内容，例如：修改日程 #3 为明天下午3点开会。"

        updated = self.dao.update_schedule(schedule["id"], **updates)
        self._last_target_schedule_id = schedule["id"]
        return f"好的，已更新日程：{self._format_item(updated)}"

    def _resolve_schedule(self, target_id: Optional[int], target_content: Optional[str]):
        if target_id:
            return self.dao.get_schedule_by_id(target_id)
        if target_content:
            rows = self.dao.find_active_by_content(target_content)
            if len(rows) == 1:
                return rows[0]
            if len(rows) > 1:
                return rows
        return None

    def _resolve_schedules_by_ids(self, target_ids: List[int]) -> tuple[List[Dict], List[int]]:
        schedules = []
        missing_ids = []
        for target_id in target_ids:
            schedule = self.dao.get_schedule_by_id(target_id)
            if schedule:
                schedules.append(schedule)
            else:
                missing_ids.append(target_id)
        return schedules, missing_ids

    def _first_pending_delete_id(self) -> Optional[int]:
        if isinstance(self.pending_delete, list):
            return self.pending_delete[0]["id"] if self.pending_delete else None
        if self.pending_delete:
            return self.pending_delete["id"]
        return None

    def _build_add_success(self, schedule_id: int, parsed: Dict) -> str:
        repeat_text = self._repeat_label(parsed)
        time_text = format_schedule_time(parsed["scheduled_time"])
        return f"好的，已添加日程：{time_text} {parsed['content']}{repeat_text}（编号 #{schedule_id}）"

    def _format_item(self, item: Dict) -> str:
        return f"#{item['id']} {format_schedule_time(item['scheduled_time'])} {item['content']}{self._repeat_label(item)}"

    def _repeat_label(self, item: Dict) -> str:
        rule = item.get("repeat_rule")
        if rule == "daily":
            return "（每天重复）"
        if rule == "weekly":
            return "（每周重复）"
        if rule == "monthly":
            return "（每月重复）"
        return ""

    def _resolve_action(self, intent: Optional[str], pending_action: Optional[str]) -> str:
        if pending_action:
            return pending_action
        return {
            "add": "add",
            "delete": "delete",
            "query": "query",
            "update": "update",
            "confirm": "confirm",
            "cancel_confirm": "cancel",
            "empty": "empty",
            "unknown": "unknown",
        }.get(intent or "unknown", "unknown")
