# 基金数据问答智能体 (NL2SQL)

## 项目概述

基于大语言模型的自然语言转SQL智能体，用于博金杯比赛基金数据问答。将用户的中文自然语言问题自动转化为SQLite SQL查询语句，对10张基金数据表进行查询并返回答案。

---

## 数据库架构 (10张表)

### 核心表

| 表名 | 描述 | 主键 | 外键 |
|------|------|------|------|
| fund_info | 基金基本信息 | fund_code | - |
| fund_stock_holdings | 基金股票持仓 | id | fund_code->fund_info |
| fund_bond_holdings | 基金债券持仓 | id | fund_code->fund_info |
| fund_convertible_bond | 基金可转债持仓 | id | fund_code->fund_info |
| fund_daily_quote | 基金日净值行情 | id | fund_code->fund_info |
| fund_scale_change | 基金规模变动 | id | fund_code->fund_info |
| fund_holder_structure | 基金持有人结构 | id | fund_code->fund_info |
| a_stock_daily | A股日行情 | (stock_code, trade_date) | stock_code->stock_industry |
| hk_stock_daily | 港股日行情 | (stock_code, trade_date) | - |
| stock_industry | 行业分类 | stock_code | - |

### 关键关联
- **fund_code** 连接基金基础信息与持仓、净值、规模等子表（1:N）
- **stock_code** 连接基金持仓与A股日行情、行业分类

### 字段说明

**fund_info**: fund_code, fund_name, fund_type(混合型/债券型/货币型/指数型), fund_manager, custodian_bank, establish_date, management_fee(%), custodian_fee(%), fund_scale(元)

**fund_stock_holdings**: fund_code, report_date(YYYYMMDD), stock_code, stock_name, hold_shares, hold_market_value(元), proportion(占净值比例%)

**fund_daily_quote**: fund_code, trade_date(YYYYMMDD), unit_net_value(单位净值), accumulated_net_value(累计净值), daily_return(日收益率%)

**fund_scale_change**: fund_code, report_period(e.g. "2021Q1"), period_start_scale, period_end_scale, subscription_share, redemption_share

**a_stock_daily**: stock_code, trade_date, stock_name, open_price, close_price, high_price, low_price, volume(成交量), turnover(成交金额), change_pct(涨跌幅%)

**stock_industry**: stock_code, stock_name, industry_level1(一级行业), industry_level2(二级行业), industry_source(分类来源)

---

## 技术架构

### 流程
```
用户问题 → [LLM(SiliconFlow)] → SQL查询 → [SQLite] → 查询结果 → [LLM] → 自然语言回答
```

### 模型
- **主模型**: deepseek-ai/DeepSeek-V3 (SiliconFlow API)
- **备选**: deepseek-ai/DeepSeek-V4-Flash, Qwen/Qwen2.5-72B-Instruct
- **温度**: 0.01 (确保SQL生成的确定性)

### 依赖
```
requests         # API调用
sqlite3          # 数据库执行（内置）
```

## 功能特性
- 自动提取数据库Schema并构建上下文
- 针对基金数据的专属prompt优化
- 常见SQL错误自动修复（重试机制）
- 增量保存结果，支持断点续跑
- JSONL格式问题批量处理

## 运行方式

### 1. 单条问答
```bash
cd /mnt/c/Users/freedom/Desktop/基金问答智能体
source venv/bin/activate
python3 code/nl2sql.py "华夏成长混合基金在20210630持有哪些股票？"
```

### 2. 批量处理
```bash
python3 code/batch_all.py
```
输出结果保存在 output/all_results.json

### 3. 交互模式
```bash
python3 code/nl2sql.py
# 进入交互式问答界面，输入exit退出
```

## 结果统计

998个DB可答问题，经测试的50个问题中：
- 成功: 48个 (96%)
- 失败: 2个 (公司法人/招股书类问题，数据库无相关信息)
- 平均处理时间: ~3秒/题

---

## 项目文件结构
```
基金问答智能体/
├── data/
│   ├── 博金杯比赛数据.db    # SQLite数据库
│   └── question.json        # 1000道问题 (JSONL格式)
├── code/
│   ├── nl2sql.py            # NL2SQL引擎
│   ├── batch_all.py         # 批量处理器
│   ├── schema.sql           # 建表DDL
│   └── generate_db.py       # 样本数据生成器
├── output/
│   ├── all_results.json     # 批量处理结果
│   ├── db_questions.json    # DB相关问题分类
│   ├── pdf_questions.json   # 招股书问题分类
│   └── summary.md           # 处理摘要
├── docs/
│   └── schema_diagram.html  # 数据库关系图
└── venv/                    # Python虚拟环境
```
