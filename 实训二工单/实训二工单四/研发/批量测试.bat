@echo off
cd /d "%~dp0"
title 批量测试

set PY=C:\Users\freedom\.conda\envs\py310\python.exe
if not exist "%PY%" set PY=D:\an1\envs\py310\python.exe
if not exist "%PY%" set PY=D:\an1\python.exe
if not exist "%PY%" set PY=python

%PY% code\batch_test.py
pause
