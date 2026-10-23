@echo off
REM One-click start of the screening application (Windows). Requires Docker Desktop.
cd /d "%~dp0"
echo Starting Trial Criterion Screening...

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not running. Start it, wait until it says "running", then run this again.
  pause
  exit /b 1
)

docker compose up -d --build
if errorlevel 1 (
  echo Failed to start the container. See the messages above.
  pause
  exit /b 1
)

echo Waiting for the server on http://localhost:8000 ...
:wait
curl -s -o nul http://localhost:8000/api/health
if errorlevel 1 (
  timeout /t 2 /nobreak >nul
  goto wait
)

start "" http://localhost:8000
echo.
echo The app is open at http://localhost:8000
echo On the first start the model is downloaded and loaded in the background (status shown top-right).
echo Stop it with:  docker compose down
pause
