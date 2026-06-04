# JMeter 压力测试说明

项目现在提供 JMeter 压力测试方案，测试文件在 `tests/jmeter/`。

## 文件说明

- `tests/jmeter/chat_load_test.jmx`：JMeter 测试计划。
- `tests/jmeter/chat_questions.csv`：聊天接口压测问题参数。
- `scripts/run_jmeter_load_test.bat`：Windows 一键运行脚本。
- `tests/jmeter/results/`：运行后生成的 `.jtl` 和 HTML 报告目录。

## 前置条件

1. 安装 Java 8 或更新版本。JMeter 必须依赖 Java 运行。
2. 安装 JMeter 5.x。
3. 当前脚本默认优先使用 `D:\tools\jmeter\apache-jmeter-5.6.3\bin\jmeter.bat`。
   如果你的路径变化，可以配置 `JMETER_HOME`，或者把 `jmeter.bat` 所在目录加入 `PATH`。
4. 先启动后端服务，例如：

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8080
```

确认健康检查可用：

```powershell
Invoke-WebRequest http://127.0.0.1:8080/health
```

确认 Java 可用：

```bat
java -version
```

如果提示找不到 Java，请安装 JDK，并设置 `JAVA_HOME`，例如：

```bat
setx JAVA_HOME "D:\tools\jdk-17"
setx PATH "%JAVA_HOME%\bin;%PATH%"
```

## 运行方式

默认参数：

```bat
scripts\run_jmeter_load_test.bat
```

自定义参数：

```bat
scripts\run_jmeter_load_test.bat 20 10 20 127.0.0.1 8080
```

参数顺序：

1. 并发线程数，默认 `10`
2. 每线程循环次数，默认 `5`
3. 启动爬坡时间秒数，默认 `10`
4. 服务 host，默认 `127.0.0.1`
5. 服务 port，默认 `8080`

## 查看结果

运行后查看：

- 原始结果：`tests/jmeter/results/chat_load_test.jtl`
- HTML 报告：`tests/jmeter/results/report/index.html`

报告里重点看：

- Throughput：吞吐量
- Average：平均响应时间
- 90% Line / 95% Line / 99% Line：百分位延迟
- Error %：错误率

## 测试内容

当前测试计划包含：

1. `GET /health`
2. `POST /api/chat`

聊天问题从 `tests/jmeter/chat_questions.csv` 读取，会覆盖医生、律师、教师、英语学习助手、金融理财师等角色。

## 和旧 Python 压测脚本的关系

`stress_test.py` 暂时保留，作为历史脚本和快速调试参考。正式压力测试请使用 JMeter：

```bat
scripts\run_jmeter_load_test.bat
```
