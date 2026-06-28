@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

C:\Users\freedom\.conda\envs\py310\python.exe code\app.py

pause
