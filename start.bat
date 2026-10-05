@echo off
chcp 65001 >nul
echo ========================================
echo   多功能AI助手 - Windows启动
echo ========================================

REM 使用 conda 环境的 python（改成你的实际路径）
set CONDA_PYTHON=D:\Users\32208\anaconda3\envs\ai-assistant\python.exe

REM 检查 .env
if not exist .env (
    echo [错误] 请先创建 .env 文件
    pause
    exit /b 1
)

REM 检查 python
"%CONDA_PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 conda Python
    pause
    exit /b 1
)

echo [启动服务...]
"%CONDA_PYTHON%" main.py

pause