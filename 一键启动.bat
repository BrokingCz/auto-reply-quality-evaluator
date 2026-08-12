@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
title CS Auto-Reply Evaluation - One-Click Runner
cd /d "%~dp0"

echo ============================================================
echo   Customer Service Auto-Reply Quality Evaluation
echo   [BONUS] One-click runner. Terminal usage is the primary
echo   workflow (see README.md for commands and pros/cons).
echo ============================================================
echo.

echo [Step 1/3] Scoring 20 auto-replies with Qwen (qwen-plus)...
python scripts\run_eval.py
if errorlevel 1 (
    echo.
    echo [ERROR] Scoring failed.
    echo   - Check that DASHSCOPE_API_KEY is set
    echo   - Or run in terminal: set DASHSCOPE_API_KEY=your_key
    echo.
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Generating final evaluation report...
python scripts\generate_report.py
if errorlevel 1 (
    echo [ERROR] Report generation failed.
    pause
    exit /b 1
)

echo.
echo [Step 3/3] Opening report...
start "" "output\eval_report.md"
echo.
echo Done. Report: output\eval_report.md
echo.
pause
