@echo off
cd /d "%~dp0"
if not exist ".env" echo [EquipMind A1] .env is missing. Run: python scripts\init_deepseek.py
call run_windows.bat
