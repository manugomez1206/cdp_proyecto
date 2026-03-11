# -*- coding: utf-8 -*-
# =============================================================================
# model_deploy.py
# Despliegue del mejor modelo como API REST con FastAPI
# =============================================================================

import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

# =============================================================================
# CARGA DEL MODELO
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mejor_modelo.pkl")

print("📂 Cargando modelo")
modelo = joblib.load(MODEL_PATH)
print(f"✅ Modelo cargado: {modelo.named_steps['model'].__class__.__name__}")

# =============================================================================
# DEFINICIÓN DE LA APP
# =============================================================================

app = FastAPI(
    title="API - Modelo de Predicción de Pago de Créditos",
    description="""
    API REST para predicción de pago oportuno de créditos.
    
    - **0** = No paga a tiempo (moroso)
    - **1** = Paga a tiempo
    
    Proyecto: CDP - Ciencia de Datos en Producción
    Autora: Manuela Gómez Gallego
    """,
    version="1.0.0"
)

# =============================================================================
# ESQUEMA DE ENTRADA
# =============================================================================

class ClienteInput(BaseModel):
    num__capital_prestado_log      : float = Field(...)
    num__plazo_meses               : float = Field(...)
    num__edad_cliente              : float = Field(...)
    num__salario_log               : float = Field(...)
    num__ratio_cuota_salario       : float = Field(...)
    num__puntaje_datacredito       : float = Field(...)
    num__huella_consulta           : float = Field(...)
    num__total_creditos            : float = Field(...)
    cat__tipo_credito_4            : float = Field(...)
    cat__tipo_credito_6            : float = Field(...)
    cat__tipo_credito_9            : float = Field(...)
    cat__tipo_credito_10           : float = Field(...)
    cat__tipo_credito_68           : float = Field(...)
    cat__tipo_laboral_Empleado     : float = Field(...)
    cat__tipo_laboral_Independiente: float = Field(...)


class BatchInput(BaseModel):
    """Schema para predicciones por batch"""
    clientes: List[ClienteInput]


class PrediccionOutput(BaseModel):
    """Schema de salida"""
    prediccion      : int
    probabilidad_0  : float
    probabilidad_1  : float
    riesgo          : str


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Endpoint raíz — verifica que la API está activa"""
    return {
        "mensaje"  : "API de Predicción de Pago de Créditos activa",
        "version"  : "1.0.0",
        "docs"     : "/docs"
    }


@app.get("/health")
def health():
    """Health check del servicio"""
    return {
        "status" : "ok",
        "modelo" : modelo.named_steps['model'].__class__.__name__
    }


@app.post("/predecir", response_model=PrediccionOutput)
def predecir(cliente: ClienteInput):
    """
    Predicción individual para un cliente.
    Retorna:
    - prediccion: 0 (no paga) o 1 (paga)
    - probabilidad_0: probabilidad de no pagar
    - probabilidad_1: probabilidad de pagar
    - riesgo: Alto / Bajo
    """
    try:
        data = pd.DataFrame([cliente.dict()])
        pred      = modelo.predict(data)[0]
        prob      = modelo.predict_proba(data)[0]

        return PrediccionOutput(
            prediccion     = int(pred),
            probabilidad_0 = round(float(prob[0]), 4),
            probabilidad_1 = round(float(prob[1]), 4),
            riesgo         = "Alto" if pred == 0 else "Bajo"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predecir/batch")
def predecir_batch(batch: BatchInput):
    """
    Predicción por batch para múltiples clientes.
    Retorna lista de predicciones.
    """
    try:
        data = pd.DataFrame([c.dict() for c in batch.clientes])

        predicciones  = modelo.predict(data)
        probabilidades = modelo.predict_proba(data)

        resultados = []
        for i, (pred, prob) in enumerate(zip(predicciones, probabilidades)):
            resultados.append({
                "cliente_id"    : i + 1,
                "prediccion"    : int(pred),
                "probabilidad_0": round(float(prob[0]), 4),
                "probabilidad_1": round(float(prob[1]), 4),
                "riesgo"        : "Alto" if pred == 0 else "Bajo"
            })

        return {
            "total_clientes"  : len(resultados),
            "total_riesgo_alto": sum(1 for r in resultados if r["riesgo"] == "Alto"),
            "total_riesgo_bajo": sum(1 for r in resultados if r["riesgo"] == "Bajo"),
            "predicciones"    : resultados
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modelo/info")
def modelo_info():
    """Información del modelo desplegado"""
    return {
        "modelo"    : modelo.named_steps['model'].__class__.__name__,
        "version"   : "1.0.0",
        "features"  : 15,
        "clases"    : {
            "0": "No paga a tiempo",
            "1": "Paga a tiempo"
        },
        "metrica_principal" : "recall_clase0",
        "recall_clase0"     : 0.488,
        "balanced_accuracy" : 0.631
    }


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )