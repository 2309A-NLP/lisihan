# -*- coding: utf-8 -*-
"""
@工单编号: 人工智能 NLP-Agent 数字人项目-日程提醒智能体任务
@作者: [AI生成]
@功能: 用户输入校验和不完整输入引导
"""

from typing import Dict, List


class InputValidator:
    """校验自然语言解析结果是否足够执行。"""

    def validate_add_payload(self, payload: Dict) -> List[str]:
        missing = []
        if not payload.get("scheduled_time"):
            missing.append("time")
        if not payload.get("content"):
            missing.append("content")
        return missing

    def build_add_prompt(self, missing: List[str]) -> str:
        if "time" in missing and "content" in missing:
            return "请告诉我要提醒的时间和事项内容，例如：今天下午5点开会。"
        if "time" in missing:
            return "请告诉我提醒时间，例如：今天下午5点、明天上午9点。"
        if "content" in missing:
            return "请告诉我要提醒你的事项内容，例如：开会、吃药、交报告。"
        return ""

    def validate_delete_payload(self, payload: Dict) -> List[str]:
        if payload.get("target_ids") or payload.get("target_id") or payload.get("target_content"):
            return []
        return ["target"]

    def build_delete_prompt(self) -> str:
        return "请告诉我要删除哪条日程，可以说“删除日程 #3”，也可以说“取消下午5点开会”。"
