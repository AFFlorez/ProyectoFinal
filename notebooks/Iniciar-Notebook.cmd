@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv. Consulta README.md para crearlo.
    pause
    exit /b 1
)
set "JUPYTER_RUNTIME_DIR=%CD%\.jupyter-local\runtime"
set "JUPYTER_CONFIG_DIR=%CD%\.jupyter-local\config"
set "IPYTHONDIR=%CD%\.jupyter-local\ipython"
if not exist ".jupyter-local\ipython" mkdir ".jupyter-local\ipython"
".venv\Scripts\python.exe" -m jupyterlab notebooks\ProyectoFinal.ipynb
if errorlevel 1 pause
endlocal
