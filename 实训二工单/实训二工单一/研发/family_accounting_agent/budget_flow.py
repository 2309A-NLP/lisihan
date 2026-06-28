class BudgetFlow:
    # 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
    def __init__(self, db, parser, formatter):
        self.db = db
        self.parser = parser
        self.fmt = formatter

    def start_set(self, text):
        month = self.parser.parse_month(text)
        amount_info = self.parser.parse_amount_info(text)
        if not amount_info:
            return self.fmt.tail("请告诉我要设置的预算金额，例如：设置本月预算5000元。"), None
        budget = amount_info["amount"]
        if budget <= 0:
            return self.fmt.tail("预算金额必须大于0，请重新输入，例如：设置本月预算5000元。"), None
        pending = {"action": "budget_confirm", "month": month, "budget": budget}
        return self.fmt.tail(f"准备设置{month}预算为{self.fmt.plain_money(budget)}元，回复“确认”后保存。"), pending

    def confirm_set(self, pending, text):
        if text != "确认":
            return self.fmt.tail("如需设置预算，请回复“确认”；如需放弃，请回复“取消”。"), pending
        self.db.set_budget(pending["month"], pending["budget"])
        return self.fmt.tail(f"已设置{pending['month']}预算为{self.fmt.plain_money(pending['budget'])}元。"), None

    def status(self, text):
        month = self.parser.parse_month(text)
        remaining = self.db.get_remaining_budget(month)
        if not remaining:
            return self.fmt.tail("您还未设置本月预算，请先设置预算，例如：‘设置本月预算5000元’")

        budget = self.fmt.plain_money(remaining["budget"])
        expense = self.fmt.plain_money(remaining["expense"])
        if remaining["remaining"] >= 0:
            rest = self.fmt.plain_money(remaining["remaining"])
            return self.fmt.tail(f"本月预算{budget}元，已支出{expense}元，剩余额度{rest}元")

        over = self.fmt.plain_money(abs(remaining["remaining"]))
        return self.fmt.tail(f"⚠️ 本月预算{budget}元，已支出{expense}元，超支{over}元，请注意控制！")
