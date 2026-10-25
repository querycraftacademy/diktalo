@echo off
rem Diktalo telepito: virtualis kornyezet + csomagok a geped hardverehez.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem --- 1) Python keresese (3.10-3.13) ---
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [HIBA] Nem talaltam Python 3.10-3.13-at. Telepitsd: https://www.python.org/downloads/
    goto :fail
)
echo Python: %PY%

rem --- 2) Virtualis kornyezet ---
if not exist ".venv\Scripts\python.exe" (
    echo Virtualis kornyezet letrehozasa (.venv^)...
    %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip >nul

rem --- 3) Alapcsomagok (OpenVINO: Intel GPU + barmilyen x86 processzor) ---
echo Alapcsomagok telepitese...
"%VPY%" -m pip install -r requirements.txt || goto :fail

rem --- 4) NVIDIA GPU eseten faster-whisper + CUDA-konyvtarak ---
where nvidia-smi >nul 2>&1
if !errorlevel! == 0 (
    echo NVIDIA GPU talalhato: faster-whisper telepitese CUDA-val...
    "%VPY%" -m pip install -r requirements-nvidia.txt || echo [FIGYELEM] A faster-whisper telepitese nem sikerult, a program processzoron fog futni.
)

rem --- 5) Beallitasok ---
if not exist "config.json" (
    copy /y "config.example.json" "config.json" >nul
    echo config.json letrehozva a config.example.json alapjan.
)

rem --- 6) Mit lat a gep? ---
echo.
"%VPY%" transcribe_file.py --detect
echo.
echo Kesz. Inditas: start.bat
echo AMD GPU eseten lasd a README "AMD GPU" szakaszat.
pause
exit /b 0

:fail
echo.
echo A telepites megszakadt.
pause
exit /b 1
