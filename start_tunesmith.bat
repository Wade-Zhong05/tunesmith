@echo off
setlocal EnableExtensions
title tunesmith
cd /d "%~dp0"

rem One-click launcher for Windows: installs pixi and the Python environment if needed,
rem downloads the trained models on the first run, starts the web app and opens it.
rem   start_tunesmith.bat       uses the GPU when an NVIDIA driver is present
rem   start_tunesmith.bat cpu   always uses the CPU
rem Set TUNESMITH_NO_BROWSER=1 to start without opening the browser.

echo.
echo   tunesmith
echo.

rem 1. pixi
where pixi >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\.pixi\bin\pixi.exe" set "PATH=%USERPROFILE%\.pixi\bin;%PATH%"
where pixi >nul 2>nul
if errorlevel 1 (
    echo [1/4] Installing pixi ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm -useb https://pixi.sh/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.pixi\bin;%PATH%"
)
where pixi >nul 2>nul
if errorlevel 1 (
    echo Could not install pixi. Install it from https://pixi.sh and run this file again.
    pause
    exit /b 1
)
echo [1/4] pixi is ready.

rem 2. Python environment
set "PIXI_ENV=default"
if /I not "%~1"=="cpu" (
    where nvidia-smi >nul 2>nul && nvidia-smi >nul 2>nul && set "PIXI_ENV=cuda"
)
if "%PIXI_ENV%"=="cuda" (
    echo [2/4] NVIDIA GPU found. Preparing the CUDA environment, which takes a while the first time ...
) else (
    echo [2/4] Preparing the CPU environment, which takes a while the first time ...
)
pixi install -e %PIXI_ENV%
if errorlevel 1 (
    echo The environment could not be installed. Check the internet connection and try again.
    pause
    exit /b 1
)

rem 3. Trained models
set "NEED_MODELS="
for %%f in ("models\composer\transformer-ref\best.pt" "models\composer\ensemble-ref\best.pt" "models\parser\lora-r16-v2\adapter_model.safetensors" "models\parser\base\Qwen2.5-0.5B-Instruct\model.safetensors" "models\soundfonts\MuseScore_General.sf3") do if not exist %%f set "NEED_MODELS=1"
if defined NEED_MODELS (
    echo [3/4] Downloading the trained models, about 1.3 GB. This only happens once ...
    pixi run -e %PIXI_ENV% download-models --all
    if errorlevel 1 echo The download did not finish. The app will still start, but may use simple baselines.
) else (
    echo [3/4] Trained models found.
)

rem 4. Start the app on the first free port from 8000
set "PORT="
for /L %%p in (8000,1,8020) do if not defined PORT (
    netstat -ano | findstr "LISTENING" | findstr /C:":%%p " >nul || set "PORT=%%p"
)
if not defined PORT set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%"
if not defined TUNESMITH_NO_BROWSER (
    start "" /min powershell -NoProfile -WindowStyle Hidden -Command "$u='%URL%'; for ($i=0; $i -lt 900; $i++) { try { Invoke-WebRequest $u -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { Start-Sleep -Seconds 1 } }"
)
echo [4/4] Starting tunesmith at %URL%
echo       The browser opens when the app is ready. Close this window to stop it.
echo.
pixi run -e %PIXI_ENV% app --port %PORT%
echo.
echo tunesmith has stopped.
pause
