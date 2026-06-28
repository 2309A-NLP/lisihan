import re
from datetime import date as date_cls
from datetime import datetime, timedelta

from constants import MEMBER_ALIASES


# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
class NlpParser:
    def is_budget_set(self, text):
        return "预算" in text and any(word in text for word in ["设置", "设定", "定个", "定"])

    def is_budget_query(self, text):
        return any(word in text for word in ["剩余额度", "预算还剩", "还剩多少", "超支"])

    def is_delete(self, text):
        return any(word in text for word in ["删除", "删掉", "去掉", "移除", "清除"])

    def is_update(self, text):
        return any(word in text for word in ["修改", "更改", "更正", "改成", "改为", "改到"])

    def is_query(self, text):
        return any(word in text for word in ["多少", "明细", "看下", "看看", "查看", "查询", "哪天", "最近", "什么", "合计", "总收入"])

    def parse_date(self, text):
        today = datetime.now().date()
        if "今天" in text:
            return today.isoformat()
        if "昨天" in text:
            return (today - timedelta(days=1)).isoformat()
        if "前天" in text:
            return (today - timedelta(days=2)).isoformat()

        for pattern in [r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?", r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"]:
            match = re.search(pattern, text)
            if match:
                return self._safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        match = re.search(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日?", text)
        if match:
            return self._safe_date(today.year, int(match.group(1)), int(match.group(2)))

        match = re.search(r"上个月\s*(\d{1,2})\s*日?", text)
        if match:
            year, month = self._last_month(today)
            return self._safe_date(year, month, int(match.group(1)))
        if "上个月" in text:
            year, month = self._last_month(today)
            return self._safe_date(year, month, 1)
        return None

    def parse_month(self, text):
        today = datetime.now().date()
        if "上个月" in text:
            year, month = self._last_month(today)
            return f"{year:04d}-{month:02d}"
        match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
        if match:
            return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
        match = re.search(r"(?<!\d)(\d{1,2})\s*月", text)
        if match and "日" not in text:
            return f"{today.year:04d}-{int(match.group(1)):02d}"
        return today.strftime("%Y-%m")

    def mentions_month(self, text):
        return any(word in text for word in ["这个月", "本月", "上个月"])

    def parse_member(self, text):
        for member, aliases in MEMBER_ALIASES.items():
            if any(alias in text for alias in aliases):
                return member
        return None

    def parse_amount_info(self, text, loose=False, last=False):
        matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(万元|万|元|块钱|块|人民币)", text))
        if not matches and loose:
            matches = list(re.finditer(r"(\d+(?:\.\d+)?)", text))
        if not matches:
            return None
        match = matches[-1] if last else matches[0]
        number = float(match.group(1))
        unit = match.group(2) if len(match.groups()) >= 2 and match.group(2) else ""
        amount = number * 10000 if "万" in unit else number
        return {"amount": round(amount, 2), "text": match.group(0)}

    def parse_amount_change(self, text):
        patterns = [
            r"从\s*(\d+(?:\.\d+)?)\s*(万元|万|元|块钱|块|人民币)?\s*改(?:成|为|到)\s*(\d+(?:\.\d+)?)\s*(万元|万|元|块钱|块|人民币)?",
            r"(\d+(?:\.\d+)?)\s*(万元|万|元|块钱|块|人民币)?\s*改(?:成|为|到)\s*(\d+(?:\.\d+)?)\s*(万元|万|元|块钱|块|人民币)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            old_number = float(match.group(1))
            old_unit = match.group(2) or ""
            new_number = float(match.group(3))
            new_unit = match.group(4) or ""
            old_amount = old_number * 10000 if "万" in old_unit else old_number
            new_amount = new_number * 10000 if "万" in new_unit else new_number
            return {
                "target_amount": round(old_amount, 2),
                "new_amount": round(new_amount, 2),
                "raw": match.group(0),
            }
        return None

    def parse_type(self, text):
        if any(word in text for word in ["收入", "收到", "报销", "工资", "奖金", "退款", "赚", "转入"]):
            return "收入"
        if any(word in text for word in ["支出", "花", "买", "付款", "支付", "付了", "交", "消费", "用了", "报名", "报了"]):
            return "支出"
        return None

    def query_type(self, text):
        if "收入" in text:
            return "收入"
        if any(word in text for word in ["花", "支出", "买", "消费"]):
            return "支出"
        return None

    def parse_item(self, text, allow_plain=False):
        result = self.remove_common_parts(text)
        result = re.sub(r"收入|支出|花了|花费|消费|用了|付了|付款|支付|买了|购买|买|收到|收了|获得|赚了|报名|报了", "", result)
        result = result.strip(" ，,。.!！？；;：:")
        if result:
            return result[:200]
        return text.strip(" ，,。.!！？；;：:")[:200] if allow_plain else None

    def parse_record_id(self, text):
        match = re.search(r"#\s*(\d+)|记录\s*(\d+)|编号\s*(\d+)", text)
        if match:
            return int(next(group for group in match.groups() if group))
        if text.strip().isdigit():
            return int(text.strip())
        return None

    def parse_update_intent(self, text):
        amount_change = self.parse_amount_change(text)
        if amount_change:
            return "amount", amount_change["new_amount"]
        if "收入" in text and re.search(r"改|修改|更正", text):
            return "type", "收入"
        if "支出" in text and re.search(r"改|修改|更正", text):
            return "type", "支出"
        if "日期" in text:
            parsed_date = self.parse_date(text)
            if parsed_date:
                return "date", parsed_date
        if "成员" in text:
            member = self.parse_member(text)
            if member:
                return "member", member
        if any(word in text for word in ["金额", "钱", "元", "块"]):
            amount = amount_change["new_amount"] if amount_change else self.parse_amount_info(text, last=True)
            if amount:
                return "amount", amount if isinstance(amount, float) else amount["amount"]
        match = re.search(r"(?:事由|项目|内容|商品).*(?:改成|改为|改到|为)(.+)$", text)
        if match:
            return "item", match.group(1).strip(" ，,。.!！？；;：:")[:200]
        return None, None

    def extract_keyword(self, text):
        result = self.remove_common_parts(text)
        stop_words = [
            "删除", "删掉", "去掉", "修改", "更改", "更正", "改成", "改为", "改到", "记录", "账目", "账",
            "看下", "看看", "查看", "查询", "明细", "哪天", "什么时候", "这个月", "本月", "上个月", "最近",
            "家里", "全家", "总收入", "收入", "支出", "花了多少钱", "花多少钱", "花了", "花钱",
            "买的", "买了", "买", "什么", "多少", "合计", "费用", "的", "把", "将", "从", "到",
        ]
        for word in stop_words:
            result = result.replace(word, "")
        return result.strip(" ，,。.!！？；;：:")

    def extract_delete_keyword(self, text):
        result = self.remove_common_parts(text)
        stop_words = [
            "我刚刚说错了", "刚刚说错了", "说错了", "我没有", "没有", "不是", "不要",
            "请帮我", "帮我", "请", "把", "将", "这条", "这笔", "这个", "一条", "一笔",
            "记录", "账目", "账", "删除", "删掉", "去掉", "移除", "清除", "的",
        ]
        for word in stop_words:
            result = result.replace(word, "")
        return result.strip(" ，,。.!！？；;：:")

    def keyword_candidates(self, keyword):
        candidates = [
            keyword,
            keyword.replace("费用", ""),
            keyword.replace("买", ""),
            keyword.replace("子", ""),
            keyword.removeprefix("报"),
            keyword.replace("报", "", 1),
        ]
        return [item for item in dict.fromkeys(item.strip() for item in candidates) if item]

    def remove_common_parts(self, text):
        result = text
        result = re.sub(r"#\s*\d+|记录\s*\d+|编号\s*\d+", "", result)
        result = re.sub(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?", "", result)
        result = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", "", result)
        result = re.sub(r"\d{1,2}\s*月\s*\d{1,2}\s*日?", "", result)
        result = re.sub(r"今天|昨天|前天|上个月|这个月|本月", "", result)
        result = re.sub(r"\d+(?:\.\d+)?\s*(万元|万|元|块钱|块|人民币)?", "", result)
        for aliases in MEMBER_ALIASES.values():
            for alias in aliases:
                result = result.replace(alias, "")
        return result

    def _safe_date(self, year, month, day):
        try:
            return date_cls(year, month, day).isoformat()
        except ValueError:
            return None

    def _last_month(self, today):
        if today.month == 1:
            return today.year - 1, 12
        return today.year, today.month - 1
