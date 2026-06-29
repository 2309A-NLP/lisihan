@echo off
title Install Auto-Start

echo ========================================
echo  Install Smart Agent Auto-Start
echo ========================================
echo.
schtasks /query /tn "SmartAgent" >nul 2>&1
if %errorlevel%==0 (
  echo [i] Auto-start already exists, removing old one...
  schtasks /delete /tn "SmartAgent" /f >nul 2>&1
)
echo.
echo Creating scheduled task...
schtasks /create /tn "SmartAgent" /tr "
  "C:\Users\freedom\.conda\envs\py310\python.exe"
  "C:\Users\freedom\Desktop\agent\06-Agent智能体项目\launcher.py"
" /sc onlogon /rl highest /delay 0000:30 /f
if %errorlevel%==0 (
  echo.
  echo [OK] Auto-start installed!
  echo      Services will start 30 seconds after you log in.
) else (
  echo.
  echo [FAILED] Please run this script as Administrator.
  echo          Right-click - Run as administrator
)
echo.
pause
