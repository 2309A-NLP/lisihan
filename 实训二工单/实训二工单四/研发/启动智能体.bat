@echo off
cd /d "%~dp0"
title 基金问答智能体

set PY=C:\Users\freedom\.conda\envs\py310\python.exe
if not exist "%PY%" set PY=D:\an1\envs\py310\python.exe
if not exist "%PY%" set PY=D:\an1\python.exe
if not exist "%PY%" set PY=python

%PY% -c "import flask,requests" 2>nul
if errorlevel 1 %PY% -m pip install flask requests -q

echo Open http://localhost:5002 in your browser
%PY% code\app.py
pause
