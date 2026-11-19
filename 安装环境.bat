@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在建立虚拟环境 .venv ...
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo 建立虚拟环境失败。请先安装 Python 3.12，见 docs\环境安装.md 第 2 节。
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -c "import torch" 2>nul
if errorlevel 1 (
  echo.
  echo 还没有安装 PyTorch。请先按 docs\环境安装.md 第 3.3 节，在本窗口里运行 PyTorch 官网给出的安装命令，
  echo 例如：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  echo 装好后再双击本文件。
  cmd /k
  exit /b 0
)
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cv-learning --display-name "Python 3 (cv-learning)"
python scripts\check_env.py
pause
