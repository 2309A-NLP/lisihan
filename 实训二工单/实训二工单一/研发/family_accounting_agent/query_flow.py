from datetime import datetime


class QueryFlow:
    # 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
    def __init__(self, db, parser, formatter):
        self.db = db
        self.parser = parser
        self.fmt = formatter

    def answer(self, text):
        member = self.parser.parse_member(text)
        month = self.parser.parse_month(text) if self.parser.mentions_month(text) else None
        query_type = self.parser.query_type(text)
        item_keyword = self.parser.extract_keyword(text)
        limit = 5 if ("最近" in text or "花了多少钱" in text or "花多少钱" in text) else 100

        if "总收入" in text or query_type == "收入":
            rows = self.db.query_records(member=member, item_keyword=item_keyword, month=month, type_="收入", limit=limit)
            return self._summary_response(self._scope_label(member, month, "收入"), rows, "收入")

        if "哪天" in text:
            rows = self.db.query_records(member=member, item_keyword=item_keyword, type_=query_type, limit=20)
            if not rows:
                return self.fmt.tail(f"没有找到包含“{item_keyword}”的记录。")
            lines = ["找到了这些相关日期："]
            lines.extend(self.fmt.format_record(row) for row in rows[:10])
            return self.fmt.tail("\n".join(lines))

        rows = self.db.query_records(member=member, item_keyword=item_keyword, month=month, type_=query_type, limit=limit)
        if "明细" in text or "看" in text or "最近" in text or "什么" in text:
            return self._detail_response(self._scope_label(member, month, query_type), rows)
        return self._summary_response(self._scope_label(member, month, query_type), rows, query_type)

    def _summary_response(self, title, rows, query_type):
        if not rows:
            return self.fmt.tail(f"没有找到{title}记录。")
        if query_type == "收入":
            total = sum(float(row["amount"]) for row in rows)
        elif query_type == "支出":
            total = sum(abs(float(row["amount"])) for row in rows)
        else:
            total = sum(float(row["amount"]) for row in rows)
        lines = [f"{title}合计{self.fmt.plain_money(total)}元。", "最近明细："]
        lines.extend(self.fmt.format_record(row) for row in rows[:5])
        return self.fmt.tail("\n".join(lines))

    def _detail_response(self, title, rows):
        if not rows:
            return self.fmt.tail(f"没有找到{title}明细。")
        lines = [f"{title}明细："]
        lines.extend(self.fmt.format_record(row) for row in rows[:15])
        return self.fmt.tail("\n".join(lines))

    def _scope_label(self, member, month, query_type):
        parts = []
        if month:
            parts.append("本月" if month == datetime.now().strftime("%Y-%m") else month)
        parts.append(member or "家里")
        if query_type:
            parts.append(query_type)
        return "".join(parts)
