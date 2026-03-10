# -*- coding: utf-8 -*-
# =============================================================================
# model_training.py
# Entrenamiento y selección del mejor modelo de créditos
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    make_scorer
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =============================================================================
# CARGA DE DATOS
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "df_train.xlsx")
TEST_PATH  = os.path.join(BASE_DIR, "df_test.xlsx")

df_train = pd.read_excel(TRAIN_PATH)
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

X_train = df_train[features]
y_train = df_train["Pago_atiempo"]
X_test  = df_test[features]
y_test  = df_test["Pago_atiempo"]

print(f"✅ Datos cargados")
print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")
print(f"🎯 Desbalance - clase 0: {(y_train==0).sum()} | clase 1: {(y_train==1).sum()}")

# =============================================================================
# FUNCIONES REQUERIDAS 
# =============================================================================

def build_model(name, estimator, X_train, y_train):
    """
    Construye un pipeline con SMOTE + modelo, lo entrena y retorna
    el pipeline entrenado junto con el nombre del modelo.

    Parámetros:
    -----------
    name      : str   — nombre del modelo
    estimator : obj   — instancia del clasificador
    X_train   : df    — features de entrenamiento
    y_train   : serie — variable objetivo de entrenamiento

    Retorna:
    --------
    name      : str      — nombre del modelo
    pipeline  : Pipeline — pipeline entrenado con SMOTE + modelo
    """
    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("model", estimator)
    ])
    pipeline.fit(X_train, y_train)
    print(f"✅ Modelo entrenado: {name}")
    return name, pipeline


def summarize_classification(name, pipeline, X_test, y_test, results):
    """
    Evalúa el modelo en test, genera el classification report y
    agrega las métricas al diccionario de resultados.

    Parámetros:
    -----------
    name     : str  — nombre del modelo
    pipeline : obj  — pipeline entrenado
    X_test   : df   — features de prueba
    y_test   : serie — variable objetivo de prueba
    results  : dict — diccionario acumulador de métricas

    Retorna:
    --------
    results  : dict — diccionario actualizado con métricas del modelo
    """
    y_pred = pipeline.predict(X_test)

    print(f"\n📋 {name} — Reporte de Clasificación:")
    print("="*55)
    print(classification_report(
        y_test, y_pred,
        target_names=["No paga (0)", "Paga (1)"]
    ))

    results[name] = {
        "recall_clase0"     : recall_score(y_test, y_pred, pos_label=0),
        "recall_clase1"     : recall_score(y_test, y_pred, pos_label=1),
        "precision_clase0"  : precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1_clase0"         : f1_score(y_test, y_pred, pos_label=0),
        "f1_clase1"         : f1_score(y_test, y_pred, pos_label=1),
        "balanced_accuracy" : balanced_accuracy_score(y_test, y_pred)
    }

    # Matriz de confusión
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No paga (0)", "Paga (1)"]
    )
    plt.title(f"Matriz de Confusión - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"))
    plt.close()

    return results

# =============================================================================
# DEFINICIÓN DE MODELOS
# =============================================================================

# Ratio de desbalance para penalizar clase minoritaria
scale = int((y_train == 1).sum() / (y_train == 0).sum())

modelos = {
    "Regresion Logistica": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        random_state=42,
        verbose=-1,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8
    )
}

# =============================================================================
# VALIDACIÓN CRUZADA
# =============================================================================

skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = {
    "recall_clase0"     : make_scorer(recall_score, pos_label=0),
    "balanced_accuracy" : "balanced_accuracy",
    "f1_clase0"         : make_scorer(f1_score, pos_label=0)
}

print("\n📊 Validación Cruzada (StratifiedKFold, 10 folds):")
print("="*55)

cv_summary = {}
for name, estimator in modelos.items():
    pipe_cv = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("model", estimator)
    ])
    cv_scores = cross_validate(
        pipe_cv, X_train, y_train,
        cv=skfold, scoring=scoring, n_jobs=-1
    )
    cv_summary[name] = {
        "recall_clase0"     : cv_scores["test_recall_clase0"].mean(),
        "balanced_accuracy" : cv_scores["test_balanced_accuracy"].mean(),
        "f1_clase0"         : cv_scores["test_f1_clase0"].mean()
    }
    print(f"\n{name}:")
    print(f"  recall_clase0     → {cv_scores['test_recall_clase0'].mean():.3f} ± {cv_scores['test_recall_clase0'].std():.3f}")
    print(f"  balanced_accuracy → {cv_scores['test_balanced_accuracy'].mean():.3f} ± {cv_scores['test_balanced_accuracy'].std():.3f}")
    print(f"  f1_clase0         → {cv_scores['test_f1_clase0'].mean():.3f} ± {cv_scores['test_f1_clase0'].std():.3f}")

# =============================================================================
# ENTRENAMIENTO Y EVALUACIÓN FINAL
# =============================================================================

pipelines = {}
results   = {}

for name, estimator in modelos.items():
    name, pipeline    = build_model(name, estimator, X_train, y_train)
    pipelines[name]   = pipeline
    results           = summarize_classification(name, pipeline, X_test, y_test, results)

# =============================================================================
# TABLA COMPARATIVA
# =============================================================================

df_results = pd.DataFrame(results).T.round(3)
df_results = df_results.sort_values("recall_clase0", ascending=False)

print("\n📊 Tabla Comparativa de Modelos:")
print("="*55)
print(df_results.to_string())

# Gráfico comparativo
df_results[["recall_clase0", "balanced_accuracy", "f1_clase0"]].plot(
    kind="bar", figsize=(12, 6), colormap="Set2"
)
plt.title("Comparación de Modelos - Métricas Clave")
plt.ylabel("Score")
plt.xlabel("Modelo")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "comparacion_modelos.png"))
plt.close()

# =============================================================================
# SELECCIÓN Y GUARDADO DEL MEJOR MODELO
# =============================================================================

mejor_modelo_nombre = df_results["recall_clase0"].idxmax()
mejor_pipeline      = pipelines[mejor_modelo_nombre]

print(f"\n Mejor modelo: {mejor_modelo_nombre}")
print(f"   recall_clase0: {df_results.loc[mejor_modelo_nombre, 'recall_clase0']:.3f}")

# Guardar modelo
MODEL_PATH = os.path.join(BASE_DIR, "mejor_modelo.pkl")
joblib.dump(mejor_pipeline, MODEL_PATH)
print(f"\n✅ Modelo guardado en: {MODEL_PATH}")