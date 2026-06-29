@echo off
title Remove Auto-Start

echo ========================================
echo  Remove Smart Agent Auto-Start
echo ========================================
echo.
schtasks /query /tn "SmartAgent" >nul 2>&1
if %errorlevel%==0 (
  schtasks /delete /tn "SmartAgent" /f >nul 2>&1
  echo [OK] Auto-start removed.
) else (
  echo [i] No auto-start task found - nothing to remove.
)
echo.
pause
