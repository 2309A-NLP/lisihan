@echo off
chcp 65001 >nul
title 文生图智能体 - 命令行模式
echo ============================================
echo  文生图智能体 - 命令行模式
echo  工单: 人工智能 NLP-Agent 数字人项目
echo ============================================
echo.
echo  请拖入人脸图片路径:
set /p INPUT=
echo.
echo  启用扩图? (Y/n):
set /p OUTPAINT=
if /i "%OUTPAINT%"=="n" (
    C:\Users\freedom\.conda\envs\py310\python.exe main.py "%INPUT%" --no-outpainting
) else (
    C:\Users\freedom\.conda\envs\py310\python.exe main.py "%INPUT%"
)
echo.
pause
