"""精确分类 v4：排除金融类公司，只保留真正的招股书问题"""
import json, os, re

INPUT = r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\question.json'
OUTPUT = r'C:\Users\freedom\Desktop\招股书问答智能体\data\招股书问题.jsonl'

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ===== 招股书强关键词 =====
PROSPECTUS_STRONG = [
    "招股", "招股说明书", "招股意向书",
    "本次发行", "发行人", "发起人", "股权转让",
    "控股股东", "占公司总股份", "占公司总股本",
    "变更设立", "变更设立时",
    "总资产周转率",
    "竞争优势", "产品研发",
    "近三年", "近三年及一期",
    "存货", "占流动资产",
    "募集资金", "募投项目",
    "前十大股东",
    "毛利率", "净利率", "产能", "产量", "销量",
    "核心技术", "研发投入",
    "主营业务收入", "营业收入构成",
    "前五大客户",
    "负责产品", "负责什么",
    "主营业务", "营业范围",
    "净资产", "总资产",
    "生产的", "公司产品",
]

# ===== 金融类公司名（基金问题，不是招股书） =====
FINANCIAL_COMPANIES = [
    "基金管理有限公司", "资产管理有限公司",
    "证券股份有限公司", "证券投资",
    "基金管理",
]

# ===== 真正的招股书公司（制造业/实业公司，从招股说明书中提问） =====
PROSPECTUS_COMPANIES = [
    "长远锂科", "沃森生物", "派思燃气", "银禧科技",
    "联化科技", "旷达汽车", "奕瑞", "博睿",
    "华润微电子", "华塑股份", "万邦生化",
    "华润化工", "大富股份", "海正",
    "双环传动", "金海环境", "新天然气", "常宝股份",
    "宏昌电子", "双星药业", "中泰股份",
]

# ===== 基金强关键词 =====
FUND_STRONG = [
    "基金代码", "基金简称", "基金管理",
    "股票代码",
    "涨跌幅", "涨停", "跌停",
    "收盘价", "开盘价", "最高价", "最低价", "成交量", "成交金额",
    "一级行业", "二级行业", "中信行业", "申万行业",
    "净申购", "净赎回",
    "资产净值", "单位净值", "累计净值",
    "基金类型", "管理人", "托管人",
    "重仓股", "持仓比例",
    "可转债", "债券",
    "机构持有", "个人持有",
    "日收益率",
    "规模变动", "持有人结构",
    "中证", "港股",
    "股票数量", "股票涨停",
    "涨了", "下跌",
]


def classify(question: str) -> str:
    q = question

    # 1. 先检查是否包含金融类公司名 → 基金问题
    for name in FINANCIAL_COMPANIES:
        if name in q:
            return "fund_db"

    # 2. 招股书强关键词
    for kw in PROSPECTUS_STRONG:
        if kw in q:
            return "prospectus"

    # 3. 已知招股书公司名
    for company in PROSPECTUS_COMPANIES:
        if company in q:
            return "prospectus"

    # 4. 基金强关键词
    for kw in FUND_STRONG:
        if kw in q:
            return "fund_db"

    # 5. 公司名模式 + 股票模式
    has_company = bool(re.search(r'[\u4e00-\u9fff]{2,}(?:股份有限公司|有限公司|股份公司|集团公司)', q))
    has_stock = "股票" in q or "基金" in q
    has_stock_code = bool(re.search(r'\d{6}', q))

    if has_company and not has_stock:
        return "prospectus"
    if has_stock or has_stock_code:
        return "fund_db"

    # 6. 兜底：含有公司名的归招股书
    if has_company:
        return "prospectus"

    return "fund_db"  # 兜底归基金


def main():
    with open(INPUT, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    prospectus = []
    fund_db = []

    for line in lines:
        d = json.loads(line)
        cat = classify(d['question'])
        item = {"id": d["id"], "question": d["question"]}
        if cat == "prospectus":
            prospectus.append(item)
        else:
            fund_db.append(item)

    print(f"总: {len(lines)}, 招股书: {len(prospectus)}, 基金: {len(fund_db)}")

    # 保存
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for item in prospectus:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存: {OUTPUT}")

    # 验证：检查被归为招股书但含基金关键词的
    print("\n=== 抽检：招股书分类中仍含基金词（可能误分）===")
    for d in prospectus:
        q = d['question']
        for kw in ['基金', '净值', '收盘价', '涨跌幅', '重仓', '净申购', '净赎回', '管理人', '托管人', '中证', '港股']:
            if kw in q:
                print(f"  id={d['id']} [{kw}]: {q[:100]}")
                break

    # 验证：检查基金分类中被误归的
    print("\n=== 抽检：基金分类中可能是招股书 ===")
    for d in fund_db:
        q = d['question']
        if ('股份有限公司' in q or '有限公司' in q) and '基金' not in q:
            # 检查是否是金融公司
            is_financial = any(f in q for f in FINANCIAL_COMPANIES)
            if not is_financial:
                print(f"  id={d['id']}: {q[:120]}")


if __name__ == "__main__":
    main()
