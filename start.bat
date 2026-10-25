@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Elobb futtasd az install.bat-ot.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
pause
