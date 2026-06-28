from constants import OPENING, TAIL


# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
class ResponseFormatter:
    def tail(self, text):
        text = text.strip()
        return text if text.endswith(TAIL) else f"{text}\n{TAIL}"

    def plain_money(self, value):
        value = round(abs(float(value or 0)), 2)
        return str(int(value)) if value.is_integer() else f"{value:.2f}"

    def display_money(self, value, record_type):
        prefix = "+" if record_type == "收入" else ""
        return f"{prefix}{self.plain_money(value)}元"

    def format_record(self, row):
        return (
            f"#{row['id']} {row['date']}，{row['member']}，{row['item']}，"
            f"{self.display_money(row['amount'], row['type'])}（{row['type']}）"
        )

    def field_label(self, field):
        return {"date": "日期", "member": "成员", "item": "事由", "amount": "金额", "type": "收支类型"}[field]

    def update_value(self, field, value):
        if field == "amount":
            return f"{self.plain_money(value)}元"
        return value

    def help_text(self):
        return (
            f"{OPENING}\n"
            "示例：\n"
            "设置本月预算5000元\n"
            "剩余额度是多少\n"
            "今天女儿买了双登山鞋499元\n"
            "7月5日妈妈收到报销1000元\n"
            "看下这个月家里花钱明细\n"
            "这个月女儿花了多少钱\n"
            "删除女儿报旅游团的费用\n"
            "把昨天买书的30元改成35元"
        )
