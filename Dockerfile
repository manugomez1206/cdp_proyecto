# =============================================================================
# Dockerfile
# Imagen para el modelo de predicción de créditos
# =============================================================================

# Imagen base
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el modelo
COPY mlops_pipeline/src/model_deploy.py .
COPY mlops_pipeline/src/mejor_modelo.pkl .

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]