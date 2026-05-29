@echo off
chcp 65001 >nul
echo ========================================
echo 多角色聊天机器人 - 启动脚本
echo ========================================
echo.

set "PYTHON_EXE=.venv\Scripts\python.exe"

rem 检查虚拟环境是否存在
if not exist "%PYTHON_EXE%" (
    echo 创建虚拟环境...
    python -m venv .venv
    echo 虚拟环境创建成功
)

rem 升级pip
echo 升级pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

rem 安装依赖
echo 安装依赖...
"%PYTHON_EXE%" -m pip install -r requirements.txt

echo.
echo ========================================
echo 依赖安装完成！
echo ========================================
echo 现在启动应用...
echo.
echo 前端页面: http://127.0.0.1:8080
echo API 文档: http://127.0.0.1:8080/docs
echo 可用接口：
echo   - GET  /health          - 健康检查
echo   - GET  /                - 前端页面
 echo   - POST /api/chat        - 聊天接口
 echo   - GET  /api/role        - 角色列表
 echo   - POST /api/user/register - 用户注册
 echo   - GET  /api/knowledge   - 知识库列表
echo.

rem 启动应用
"%PYTHON_EXE%" -m uvicorn main:app --reload --port 8080
