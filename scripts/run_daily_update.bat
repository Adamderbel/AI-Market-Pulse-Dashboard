@echo off
REM Simple batch file for automated daily market data update
REM This can be used with Windows Task Scheduler

REM Set the project directory (adjust this path to your actual project location)
set PROJECT_DIR=d:\AI automation

REM Change to the project directory
cd /d "%PROJECT_DIR%"

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Run the simplified automated update script
python scripts\auto_daily_update.py

REM Log the exit code
echo Update completed with exit code: %ERRORLEVEL%

REM Optional: Add timestamp to log
echo Update finished at %DATE% %TIME%

REM Exit with the same code as the Python script
exit /b %ERRORLEVEL%
