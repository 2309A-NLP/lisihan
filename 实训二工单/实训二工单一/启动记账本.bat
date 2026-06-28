@echo off
chcp 65001 >nul
title 家庭记账助手
cd /d "C:\Users\freedom\Desktop\agent\实训二工单一\family_accounting_agent"

echo =============================
echo   家庭记账助手
echo =============================
echo.

:: --- MySQL 密码（改这里） ---
set MYSQL_PASSWORD=root

echo [1/3] 启动 MySQL ...
net start MySQL80 >nul 2>&1
if %errorlevel% neq 0 (
    net start MySQL >nul 2>&1
)
echo.

echo [2/3] 安装依赖...
C:\Users\freedom\.conda\envs\py310\python.exe -m pip install -r requirements.txt -q
echo.

echo [3/3] 启动服务...
echo.
echo   访问地址: http://127.0.0.1:8081
echo.
start "" http://127.0.0.1:8081
C:\Users\freedom\.conda\envs\py310\python.exe app.py

echo.
echo 服务已停止
pause
