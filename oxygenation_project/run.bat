@echo off
echo Запуск MLflow-сервера...
start cmd /k "mlflow ui --port 5000"

timeout /t 5 >nul

echo Запуск FastAPI backend...
start cmd /k "uvicorn backend.main:app --reload"

timeout /t 5 >nul

echo Запуск десктоп-приложения...
python desktop_ui/app.py
