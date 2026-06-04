@echo off
setlocal

set DEFAULT_JMETER_BIN=D:\tools\jmeter\apache-jmeter-5.6.3\bin\jmeter.bat
set JMETER_BIN=%DEFAULT_JMETER_BIN%
if not exist "%JMETER_BIN%" (
  set JMETER_BIN=%JMETER_HOME%\bin\jmeter.bat
)
if not exist "%JMETER_BIN%" (
  set JMETER_BIN=jmeter
)

set BASE_DIR=%~dp0..
cd /d "%BASE_DIR%"

if defined JAVA_HOME (
  if exist "%JAVA_HOME%\bin\java.exe" set "PATH=%JAVA_HOME%\bin;%PATH%"
)

where java >nul 2>nul
if errorlevel 1 (
  echo Java not found. JMeter requires Java 8 or newer.
  echo Please install Java and set JAVA_HOME, or add java.exe to PATH.
  echo Example JAVA_HOME: D:\tools\jdk-17
  exit /b 2
)

if not exist tests\jmeter\results mkdir tests\jmeter\results
if exist tests\jmeter\results\report rmdir /s /q tests\jmeter\results\report
if exist tests\jmeter\results\chat_load_test.jtl del /q tests\jmeter\results\chat_load_test.jtl

set THREADS=%1
if "%THREADS%"=="" set THREADS=10

set LOOPS=%2
if "%LOOPS%"=="" set LOOPS=5

set RAMP_UP=%3
if "%RAMP_UP%"=="" set RAMP_UP=10

set HOST=%4
if "%HOST%"=="" set HOST=127.0.0.1

set PORT=%5
if "%PORT%"=="" set PORT=8080

echo Running JMeter load test: threads=%THREADS% loops=%LOOPS% ramp_up=%RAMP_UP% host=%HOST% port=%PORT%
echo JMeter: %JMETER_BIN%

"%JMETER_BIN%" -n ^
  -t tests\jmeter\chat_load_test.jmx ^
  -l tests\jmeter\results\chat_load_test.jtl ^
  -e -o tests\jmeter\results\report ^
  -Jthreads=%THREADS% ^
  -Jloops=%LOOPS% ^
  -Jramp_up=%RAMP_UP% ^
  -Jhost=%HOST% ^
  -Jport=%PORT%

echo JMeter report: tests\jmeter\results\report\index.html
endlocal
