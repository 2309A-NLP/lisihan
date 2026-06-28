# 日程提醒智能体

日程提醒智能体是一个基于自然语言的日程管理助手，支持添加、查询、删除、修改、循环提醒、Web 聊天界面和后台提醒。

## 功能特性

- 自然语言添加日程
- 查询今天、明天、全部和最近日程
- 删除日程并进行二次确认
- 修改日程时间和内容
- 支持每天、每周、每月循环提醒
- Web 聊天界面
- 后台守护进程提醒
- MySQL 数据存储
- Agent 执行日志记录

## 快速开始

1. 安装依赖。

```bash
pip install -r requirements.txt
```

2. 配置数据库。

编辑 `config.py`，填写 MySQL 用户名、密码、端口和数据库名。

3. 初始化数据库。

```bash
mysql -u root -p < db/schema.sql
```

4. 启动 Web 界面。

```bash
python web_app.py
```

也可以双击 `start_web.bat`。

5. 开始对话。

在浏览器中打开 `http://localhost:5000`，输入自然语言日程指令。

## 项目结构

```text
daily_scheduler_agent/
├── agent/
│   ├── intent_parser.py
│   └── scheduler_agent.py
├── db/
│   ├── mysql_connector.py
│   ├── schedule_dao.py
│   ├── execution_log_dao.py
│   └── schema.sql
├── reminder/
│   ├── reminder_service.py
│   └── message_templates.py
├── templates/
│   └── index.html
├── utils/
│   ├── time_utils.py
│   ├── validator.py
│   └── logger.py
├── logs/
│   └── agent.log
├── config.py
├── interactive.py
├── scheduler_daemon.py
├── web_app.py
├── start_web.bat
├── manage.sh
├── requirements.txt
├── skill.md
└── README.md
```

## 测试用例

| 输入 | 期望输出 |
| --- | --- |
| 下午5点开会 | 已添加日程今天17点开会 |
| 我今天有什么日程 | 日程列表 |
| 每天上午8点起床 | 已添加循环日程 |
| 提醒我买咖啡 | 反问时间 |
| 删除日程1 | 确认删除 |

## 数据库验证 SQL

```sql
SELECT COUNT(*) AS schedules_count FROM schedules;
SELECT COUNT(*) AS agent_execution_logs_count FROM agent_execution_logs;
SELECT COUNT(*) AS reminder_logs_count FROM reminder_logs;
SELECT id, content, scheduled_time, repeat_rule, status FROM schedules ORDER BY id DESC LIMIT 10;
SELECT id, user_input, intent, action, result, execution_time FROM agent_execution_logs ORDER BY execution_time DESC LIMIT 10;
```

## 工单信息

工单编号：人工智能 NLP-Agent 数字人项目-日程提醒智能体任务。

作者：[AI生成]。

版本：1.0.0。
