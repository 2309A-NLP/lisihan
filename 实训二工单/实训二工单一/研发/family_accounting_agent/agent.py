from add_flow import AddRecordFlow
from budget_flow import BudgetFlow
from constants import CANCEL_WORDS, END_WORDS, OPENING
from db import DB
from formatters import ResponseFormatter
from mutation_flow import MutationFlow
from nlp import NlpParser
from query_flow import QueryFlow


# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
class Agent:
    def __init__(self):
        self.db = DB()
        self.parser = NlpParser()
        self.fmt = ResponseFormatter()
        self.add_flow = AddRecordFlow(self.db, self.parser, self.fmt)
        self.budget_flow = BudgetFlow(self.db, self.parser, self.fmt)
        self.query_flow = QueryFlow(self.db, self.parser, self.fmt)
        self.mutation_flow = MutationFlow(self.db, self.parser, self.fmt)
        self.pending = None

    def opening(self):
        return OPENING

    def process(self, text):
        text = (text or "").strip()
        normalized_text = self._normalize_short_text(text)
        if not text:
            return self.fmt.tail(self.opening())

        if normalized_text in END_WORDS:
            self.pending = None
            return "好的，有需要随时叫我。"

        if normalized_text in CANCEL_WORDS:
            self.pending = None
            return "已取消操作。"

        if self.pending:
            if text in {"确认", "确认删除", "确认修改", "确认金额", "是"}:
                return self._process_pending(text)
            if self.parser.is_delete(text):
                self.pending = None
                return self._set_pending(*self.mutation_flow.start_delete(text))
            return self._process_pending(text)

        if text.lower() in {"help", "h"} or text in {"帮助", "怎么用", "你好", "您好"}:
            return self.fmt.tail(self.fmt.help_text())
        if self.parser.is_budget_set(text):
            return self._set_pending(*self.budget_flow.start_set(text))
        if self.parser.is_budget_query(text):
            return self.budget_flow.status(text)
        if self.parser.is_delete(text):
            return self._set_pending(*self.mutation_flow.start_delete(text))
        if self.parser.is_update(text):
            return self._set_pending(*self.mutation_flow.start_update(text))
        if self.parser.is_query(text):
            return self.query_flow.answer(text)
        return self._set_pending(*self.add_flow.start(text))

    def _process_pending(self, text):
        normalized_text = self._normalize_short_text(text)
        if normalized_text in END_WORDS:
            self.pending = None
            return "好的，有需要随时叫我。"
        if normalized_text in CANCEL_WORDS:
            self.pending = None
            return "已取消。本次没有改动数据库。"

        action = self.pending.get("action")
        handlers = {
            "budget_confirm": self.budget_flow.confirm_set,
            "add_collect": self.add_flow.collect_field,
            "add_amount_confirm": self.add_flow.confirm_amount,
            "add_confirm": self.add_flow.confirm_save,
            "delete_choose": self.mutation_flow.choose_delete,
            "delete_confirm": self.mutation_flow.confirm_delete,
            "update_choose": self.mutation_flow.choose_update,
            "update_confirm": self.mutation_flow.confirm_update,
        }
        handler = handlers.get(action)
        if not handler:
            self.pending = None
            return self.fmt.tail(self.opening())

        reply, next_pending = handler(self.pending, text)
        self.pending = next_pending
        return reply

    def _set_pending(self, reply, pending):
        self.pending = pending
        return reply

    def _normalize_short_text(self, text):
        return text.strip(" \t\r\n，,。.!！?？；;：:")

    def close(self):
        self.db.close()
