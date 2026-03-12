@echo off
echo ========================================
echo  CDP - Setup Proyecto Creditos
echo  Autora: Manuela Gomez Gallego
echo ========================================

echo.
echo [1/4] Creando entorno virtual...
python -m venv .venv

echo.
echo [2/4] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo.
echo [3/4] Instalando dependencias...
pip install -r requirements.txt

echo.
echo [4/4] Setup completado!
echo.
echo Para correr el proyecto:
echo   Terminal 1: python mlops_pipeline/src/model_deploy.py
echo   Terminal 2: streamlit run mlops_pipeline/src/model_monitoring.py
echo.
pause