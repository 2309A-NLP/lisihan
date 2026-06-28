# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
from constants import FIELD_ORDER, FIELD_PROMPTS
# FIELD_ORDER:字段收集顺序
# FIELD_PROMPTS：每个字段对应的提问文案

class AddRecordFlow:

    def __init__(self, db, parser, formatter):
        self.db = db   # 数据库操作对象
        self.parser = parser # 文本分析器（提取日期、金额等）
        self.fmt = formatter # 格式化器（美化输出文本）

    def start(self, text):
        record = self._extract_record(text)  # 从用户输入提取初始信息
        return self._next_step(record)  # 进入下一步状态机

    def collect_field(self, pending, text):
        record = pending["record"]  # 获取当前记录
        field = pending["field"]  # 当前要收集的字段名
        self._fill_field(record, field, text, loose_amount=True)  # 填充字段值
        if record.get(field) in (None, ""):
            return self.fmt.tail(FIELD_PROMPTS[field]), pending
        return self._next_step(record)

    def confirm_amount(self, pending, text):
        record = pending["record"]
        if text == "确认金额":
            record["_amount_confirmed"] = True
            return self._next_step(record)

        amount_info = self.parser.parse_amount_info(text, loose=True)
        if not amount_info:
            return self.fmt.tail("请回复“确认金额”，或重新输入正确金额，比如：499元。"), pending
        record["amount"] = amount_info["amount"]
        record["_amount_text"] = amount_info["text"]
        record["_amount_confirmed"] = False
        return self._next_step(record)

    def confirm_save(self, pending, text):
        if text != "确认":
            return self.fmt.tail("如需保存，请回复“确认”；如需放弃，请回复“取消”。"), pending

        record = pending["record"]
        record_id = self.db.add_record(record["date"], record["member"], record["item"], record["amount"], record["type"])
        amount_text = self.fmt.display_money(record["amount"], record["type"])
        reply = (
            f"已记账：{record['date']}，{record['member']}，{record['item']}，"
            f"{amount_text}（{record['type']}）。记录编号#{record_id}"
        )
        return self.fmt.tail(reply), None

    def _next_step(self, record):
        for field in FIELD_ORDER:
            if record.get(field) in (None, ""):
                pending = {"action": "add_collect", "record": record, "field": field}
                return self.fmt.tail(FIELD_PROMPTS[field]), pending

        if self._needs_amount_confirm(record) and not record.get("_amount_confirmed"):
            pending = {"action": "add_amount_confirm", "record": record}
            reply = (
                f"金额为{self.fmt.plain_money(record['amount'])}元，数额为0或较大，请二次确认。"
                "如果无误请回复“确认金额”；如果有误请重新输入金额。"
            )
            return self.fmt.tail(reply), pending

        pending = {"action": "add_confirm", "record": record}
        reply = (
            "请确认记账信息：\n"
            f"日期：{record['date']}\n"
            f"成员：{record['member']}\n"
            f"事由：{record['item']}\n"
            f"金额：{self.fmt.display_money(record['amount'], record['type'])}\n"
            f"收支类型：{record['type']}\n"
            "回复“确认”保存，或回复“取消”取消。"
        )
        return self.fmt.tail(reply), pending

    def _extract_record(self, text):
        amount_info = self.parser.parse_amount_info(text)
        return {
            "date": self.parser.parse_date(text),
            "member": self.parser.parse_member(text),
            "item": self.parser.parse_item(text),
            "amount": amount_info["amount"] if amount_info else None,
            "type": self.parser.parse_type(text),
            "_amount_text": amount_info["text"] if amount_info else "",
            "_amount_confirmed": False,
        }

    def _fill_field(self, record, field, text, loose_amount=False):
        if field == "date":
            record["date"] = self.parser.parse_date(text)
        elif field == "member":
            record["member"] = self.parser.parse_member(text)
        elif field == "item":
            record["item"] = self.parser.parse_item(text, allow_plain=True)
        elif field == "amount":
            amount_info = self.parser.parse_amount_info(text, loose=loose_amount)
            if amount_info:
                record["amount"] = amount_info["amount"]
                record["_amount_text"] = amount_info["text"]
        elif field == "type":
            record["type"] = self.parser.parse_type(text)

    def _needs_amount_confirm(self, record):
        amount = float(record["amount"])
        return amount == 0 or amount >= 1_000_000 or ("万" in record.get("_amount_text", "") and amount >= 100_000)
