class MutationFlow:
    # 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
    def __init__(self, db, parser, formatter):
        self.db = db
        self.parser = parser
        self.fmt = formatter

    def start_delete(self, text):
        records = self._locate_records(text, delete_mode=True)
        if not records:
            return self.fmt.tail("没有找到匹配记录。请提供记录编号或更明确的事由，比如：删除#12、删掉登山鞋。"), None
        if len(records) == 1:
            pending = {"action": "delete_confirm", "records": records}
            return self.fmt.tail(f"找到这条记录：\n{self.fmt.format_record(records[0])}\n确认删除吗？（回复‘确认删除’或‘取消’）"), pending
        pending = {"action": "delete_choose", "records": records}
        lines = [f"找到{len(records)}条匹配记录："]
        lines.extend(self.fmt.format_record(row) for row in records[:10])
        lines.append("请指定要删除的记录编号")
        return self.fmt.tail("\n".join(lines)), pending

    def choose_delete(self, pending, text):
        record_id = self.parser.parse_record_id(text)
        ids = {row["id"] for row in pending["records"]}
        if record_id not in ids:
            return self.fmt.tail("请从上面列出的记录编号里选择一个，或回复“取消”。"), pending
        record = next(row for row in pending["records"] if row["id"] == record_id)
        next_pending = {"action": "delete_confirm", "records": [record]}
        return self.fmt.tail(f"将删除这条记录：\n{self.fmt.format_record(record)}\n确认删除吗？（回复‘确认删除’或‘取消’）"), next_pending

    def confirm_delete(self, pending, text):
        if text not in {"确认删除", "是", "确认"}:
            return self.fmt.tail("如需删除，请回复“确认删除”；如需放弃，请回复“取消”。"), pending
        deleted = []
        for row in pending["records"]:
            if self.db.delete_record(row["id"]):
                deleted.append(f"已删除记录#{row['id']}")
        return self.fmt.tail("\n".join(deleted or ["没有记录被删除。"])), None

    def start_update(self, text):
        field, value = self.parser.parse_update_intent(text)
        if not field:
            return self.fmt.tail("请告诉我要修改哪个字段，比如：把#7的事由改成买菜，或把昨天买书的30元改成35元。"), None
        amount_change = self.parser.parse_amount_change(text)
        amount_hint = amount_change["target_amount"] if amount_change else None
        if amount_change and field == "amount":
            value = amount_change["new_amount"]
        records = self._locate_records(text, amount_hint=amount_hint)
        if not records:
            return self.fmt.tail("没有找到要修改的记录。请提供记录编号或更明确的日期、成员、事由。"), None
        if len(records) == 1:
            pending = {"action": "update_confirm", "record": records[0], "field": field, "value": value}
            return self.fmt.tail(self._update_confirm_text(records[0], field, value)), pending
        pending = {"action": "update_choose", "records": records, "field": field, "value": value}
        lines = ["找到多条匹配记录，请回复要修改的记录编号："]
        lines.extend(self.fmt.format_record(row) for row in records[:10])
        return self.fmt.tail("\n".join(lines)), pending

    def choose_update(self, pending, text):
        record_id = self.parser.parse_record_id(text)
        ids = {row["id"] for row in pending["records"]}
        if record_id not in ids:
            return self.fmt.tail("请从上面列出的记录编号里选择一个，或回复“取消”。"), pending
        record = next(row for row in pending["records"] if row["id"] == record_id)
        next_pending = {"action": "update_confirm", "record": record, "field": pending["field"], "value": pending["value"]}
        return self.fmt.tail(self._update_confirm_text(record, pending["field"], pending["value"])), next_pending

    def confirm_update(self, pending, text):
        if text != "确认修改":
            return self.fmt.tail("如需修改，请回复“确认修改”；如需放弃，请回复“取消”。"), pending
        record_id = pending["record"]["id"]
        self.db.update_record(record_id, pending["field"], pending["value"])
        reply = f"已将记录#{record_id}的{self.fmt.field_label(pending['field'])}修改为{self.fmt.update_value(pending['field'], pending['value'])}"
        return self.fmt.tail(reply), None

    def _update_confirm_text(self, record, field, value):
        return (
            f"定位到记录：\n{self.fmt.format_record(record)}\n"
            f"准备把{self.fmt.field_label(field)}修改为：{self.fmt.update_value(field, value)}\n"
            "确认修改吗？（回复‘确认修改’或‘取消’）"
        )

    def _locate_records(self, text, delete_mode=False, amount_hint=None):
        record_id = self.parser.parse_record_id(text)
        if record_id:
            row = self.db.get_record_by_id(record_id)
            return [row] if row else []

        member = self.parser.parse_member(text)
        item_keyword = self.parser.extract_delete_keyword(text) if delete_mode else self.parser.extract_keyword(text)
        parsed_date = self.parser.parse_date(text)
        amount_info = self.parser.parse_amount_info(text)
        rows = self.db.query_records(
            member=member,
            item_keyword=item_keyword,
            start_date=parsed_date,
            end_date=parsed_date,
            limit=30,
        )

        target_amount = amount_hint if amount_hint is not None else (round(float(amount_info["amount"]), 2) if amount_info else None)
        if target_amount is not None:
            target = round(float(target_amount), 2)
            amount_filtered = [row for row in rows if round(abs(float(row["amount"])), 2) == target]
            if amount_filtered:
                return amount_filtered
        if rows:
            return rows

        for candidate in self.parser.keyword_candidates(item_keyword):
            rows = self.db.query_records(member=member, item_keyword=candidate, limit=30)
            if target_amount is not None:
                rows = [row for row in rows if round(abs(float(row["amount"])), 2) == round(float(target_amount), 2)]
            if rows:
                return rows
        if member:
            rows = self.db.query_records(member=member, limit=10)
            if target_amount is not None:
                rows = [row for row in rows if round(abs(float(row["amount"])), 2) == round(float(target_amount), 2)]
            return rows
        return []
