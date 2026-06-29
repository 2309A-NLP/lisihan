@echo off
title Agent CLI (MCP)

set PROJ_DIR=%~dp0
"C:\Users\freedom\.conda\envs\py310\python.exe" "%PROJ_DIR%core\agent.py"
