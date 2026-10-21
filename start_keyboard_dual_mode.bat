@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Offline usage: start_keyboard_dual_mode.bat --offline-session "session.zip" --offline-output "output"
cd /d "%~dp0" || exit /b 10
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m ssvep_fbcca --fbcca-profile m3_5band --protocol standard_5s --response-delay 0.23 %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.10 -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            py -3.10 -m ssvep_fbcca --fbcca-profile m3_5band --protocol standard_5s --response-delay 0.23 %*
        ) else (
            py -3 -m ssvep_fbcca --fbcca-profile m3_5band --protocol standard_5s --response-delay 0.23 %*
        )
    ) else (
        python -m ssvep_fbcca --fbcca-profile m3_5band --protocol standard_5s --response-delay 0.23 %*
    )
)
set "EXIT_CODE=%ERRORLEVEL%"
echo Exit code: %EXIT_CODE%
if not "%FBCCA_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%
