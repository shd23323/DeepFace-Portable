@echo off
chcp 65001 >nul
title DeepFace Demo (Portable)
setlocal
set "PYTHONIOENCODING=utf-8"
set "DEEPFACE_HOME=%~dp0models"
"%~dp0python\python.exe" "%~dp0main.py"
echo.
echo Press any key to close this window...
pause >nul
endlocal
