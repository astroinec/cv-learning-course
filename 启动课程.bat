@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo 还没有建立环境，先双击 安装环境.bat。
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m jupyter lab .
