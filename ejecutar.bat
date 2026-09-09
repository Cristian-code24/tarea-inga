@echo off
:: ================================================================
:: ejecutar.bat - Calculo III - Carlos Inga
:: Lanzador portable USB para Windows
:: ================================================================
title Calculo III - Carlos Inga - UNJFSC

echo.
echo  ================================================================
echo    CALCULO III - Funciones de Varias Variables
echo    Carlos Inga - UNJFSC
echo    Iniciando aplicacion Shiny for Python...
echo  ================================================================
echo.

:: Verificar Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python no fue encontrado en el sistema.
    echo.
    echo  Soluciones:
    echo  1. Descarga Python 3.10+ desde https://python.org/downloads
    echo  2. Durante la instalacion marca "Add Python to PATH"
    echo  3. Reinicia y vuelve a ejecutar este archivo
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  Python: %%v
echo.

:: Entorno virtual relativo al .bat
set "ROOT=%~dp0"
set "VENV=%ROOT%venv_calculo3"

:: Crear venv si no existe
if not exist "%VENV%\Scripts\activate.bat" (
    echo  [1/3] Creando entorno virtual en: %VENV%
    python -m venv "%VENV%"
    if %errorlevel% neq 0 (
        echo  [ERROR] No se pudo crear el entorno virtual.
        pause & exit /b 1
    )
    echo        OK - Entorno virtual creado.
    echo.
) else (
    echo  [1/3] Entorno virtual encontrado. Reutilizando...
    echo.
)

:: Activar venv
call "%VENV%\Scripts\activate.bat"

:: Actualizar pip silenciosamente
python -m pip install --upgrade pip --quiet --disable-pip-version-check >nul 2>&1

:: Instalar dependencias
echo  [2/3] Instalando dependencias (puede tomar 1-3 min la 1ra vez)...
python -m pip install -r "%ROOT%requirements.txt" --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo  [ERROR] Fallo al instalar dependencias.
    echo  Verifica tu conexion a internet e intenta de nuevo.
    pause & exit /b 1
)
echo        OK - Dependencias instaladas.
echo.

:: Lanzar app Shiny
echo  [3/3] Lanzando Calculo III en http://localhost:8080
echo.
echo  Abriendo navegador automaticamente en 3 segundos...
echo  Para detener la app cierra esta ventana o presiona Ctrl+C
echo.

:: Abrir navegador en paralelo (espera 3s para que el servidor arranque)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8080"

:: Cambiar al directorio del proyecto y correr
cd /d "%ROOT%"
python -m shiny run app.py --host 0.0.0.0 --port 8080

:: Mensaje de cierre
echo.
echo  ================================================
echo    La aplicacion se cerro.
echo  ================================================
pause >nul
