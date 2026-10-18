@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv. Consulta README.md para crearlo.
    pause
    exit /b 1
)
set "JUPYTER_RUNTIME_DIR=%~dp0.jupyter-local\runtime"
set "JUPYTER_CONFIG_DIR=%~dp0.jupyter-local\config"
set "IPYTHONDIR=%~dp0.jupyter-local\ipython"
if not exist ".jupyter-local\ipython" mkdir ".jupyter-local\ipython"
".venv\Scripts\python.exe" -m jupyterlab ProyectoFinal.ipynb
if errorlevel 1 pause
endlocal
