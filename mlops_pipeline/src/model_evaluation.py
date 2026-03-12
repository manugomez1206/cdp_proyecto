# -*- coding: utf-8 -*-
# =============================================================================
# model_evaluation.py
# Evaluación del mejor modelo desplegado
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score
)

# =============================================================================
# CARGA DEL MODELO Y DATOS
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "mejor_modelo.pkl")
TEST_PATH   = os.path.join(BASE_DIR, "df_test.xlsx")

print("📂 Cargando modelo y datos")
modelo   = joblib.load(MODEL_PATH)
df_test  = pd.read_excel(TEST_PATH)

features = [
    "num__capital_prestado_log",
    "num__plazo_meses",
    "num__edad_cliente",
    "num__salario_log",
    "num__ratio_cuota_salario",
    "num__puntaje_datacredito",
    "num__huella_consulta",
    "num__total_creditos",
    "cat__tipo_credito_4",
    "cat__tipo_credito_6",
    "cat__tipo_credito_9",
    "cat__tipo_credito_10",
    "cat__tipo_credito_68",
    "cat__tipo_laboral_Empleado",
    "cat__tipo_laboral_Independiente"
]

X_test = df_test[features]
y_test = df_test["Pago_atiempo"]

print(f"✅ Modelo cargado: {modelo.named_steps['model'].__class__.__name__}")
print(f"📊 Test: {X_test.shape}")
print(f"🎯 Desbalance test - clase 0: {(y_test==0).sum()} | clase 1: {(y_test==1).sum()}")

# =============================================================================
# PREDICCIONES
# =============================================================================

y_pred      = modelo.predict(X_test)
y_prob      = modelo.predict_proba(X_test)[:, 0]  # probabilidad clase 0

# =============================================================================
# MÉTRICAS DETALLADAS
# =============================================================================

print("\n📋 Reporte de Clasificación:")
print("="*55)
print(classification_report(
    y_test, y_pred,
    target_names=["No paga (0)", "Paga (1)"]
))

# Tabla resumen de métricas
metricas = {
    "Recall clase 0 (morosos)"      : recall_score(y_test, y_pred, pos_label=0),
    "Recall clase 1 (buenos)"       : recall_score(y_test, y_pred, pos_label=1),
    "Precision clase 0"             : precision_score(y_test, y_pred, pos_label=0, zero_division=0),
    "Precision clase 1"             : precision_score(y_test, y_pred, pos_label=1, zero_division=0),
    "F1 clase 0"                    : f1_score(y_test, y_pred, pos_label=0),
    "F1 clase 1"                    : f1_score(y_test, y_pred, pos_label=1),
    "Balanced Accuracy"             : balanced_accuracy_score(y_test, y_pred),
    "ROC AUC"                       : roc_auc_score(y_test, y_prob),
    "Average Precision (clase 0)"   : average_precision_score(y_test, y_prob, pos_label=0)
}

df_metricas = pd.DataFrame(
    metricas.items(),
    columns=["Métrica", "Valor"]
).round({"Valor": 3})

print("\n📊 Tabla de Métricas:")
print("="*55)
print(df_metricas.to_string(index=False))

#=============================================================================
# GUARDAR MÉTRICAS EN EXCEL
#=============================================================================
METRICS_PATH = os.path.join(BASE_DIR, "metricas_modelo.xlsx")
df_metricas.to_excel(METRICS_PATH, index=False)
print(f"\n✅ Métricas guardadas en: {METRICS_PATH}")

# =============================================================================
# GRÁFICAS DE EVALUACIÓN
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Evaluación del Modelo - Regresión Logística", fontsize=14, fontweight="bold")

# 1. Matriz de confusión
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["No paga (0)", "Paga (1)"],
    ax=axes[0, 0],
    colorbar=False
)
axes[0, 0].set_title("Matriz de Confusión")

# 2. Curva ROC
RocCurveDisplay.from_predictions(
    y_test, y_prob,
    pos_label=0,
    ax=axes[0, 1],
    name="Regresión Logística"
)
axes[0, 1].set_title("Curva ROC - clase 0 (morosos)")

# 3. Curva Precision-Recall
PrecisionRecallDisplay.from_predictions(
    y_test, y_prob,
    pos_label=0,
    ax=axes[1, 0],
    name="Regresión Logística"
)
axes[1, 0].set_title("Curva Precision-Recall - clase 0")

# 4. Distribución de probabilidades
axes[1, 1].hist(
    y_prob[y_test == 0], bins=30, alpha=0.6,
    color="red", label="No paga (0)"
)
axes[1, 1].hist(
    y_prob[y_test == 1], bins=30, alpha=0.6,
    color="blue", label="Paga (1)"
)
axes[1, 1].set_title("Distribución de Probabilidades")
axes[1, 1].set_xlabel("Probabilidad clase 0")
axes[1, 1].set_ylabel("Frecuencia")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "evaluacion_modelo.png"))
plt.close()

print(f"Gráficas guardadas en: evaluacion_modelo.png")
print("\n Evaluación completada")