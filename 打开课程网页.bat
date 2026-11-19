@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
start "" http://localhost:8000/site/index.html
%PY% -m http.server 8000
