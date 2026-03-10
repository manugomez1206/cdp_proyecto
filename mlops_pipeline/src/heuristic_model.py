# -*- coding: utf-8 -*-
# =============================================================================
# heuristic_model.py
# Modelo Heurístico Baseline para predicción de pago de créditos
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    ShuffleSplit
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    make_scorer,
    recall_score
)

# =============================================================================
# CARGA DE DATOS
# =============================================================================

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "df_train.xlsx")
TEST_PATH   = os.path.join(BASE_DIR, "df_test.xlsx")

df_train = pd.read_excel(TRAIN_PATH)
df_test  = pd.read_excel(TEST_PATH)

# Features — columnas generadas por ft_engineering.py
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
print(f"🎯 Desbalance train - clase 0: {(y_train==0).sum()} | clase 1: {(y_train==1).sum()}")

# =============================================================================
# MODELO HEURÍSTICO
# =============================================================================

class HeuristicModel(BaseEstimator, ClassifierMixin):
    """
    Modelo basado en reglas de negocio para detectar clientes
    que NO pagarán a tiempo (clase 0).

    Reglas:
      - ratio_cuota_salario alto  → riesgo de no pago
      - puntaje_datacredito bajo  → riesgo de no pago
      - huella_consulta alta      → riesgo de no pago
      - tipo_laboral Independiente → mayor riesgo
    Si tiene 2 o más señales → predice 0 (no paga)
    """

    def __init__(
        self,
        ratio_threshold=0.25,
        credit_score_threshold=0.0,
        consultas_threshold=0.5
    ):
        self.ratio_threshold          = ratio_threshold
        self.credit_score_threshold   = credit_score_threshold
        self.consultas_threshold      = consultas_threshold

    def fit(self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        predictions = []

        for _, row in X.iterrows():
            riesgo = 0

            # Cuota muy alta respecto al salario
            if row["num__ratio_cuota_salario"] > self.ratio_threshold:
                riesgo += 1

            # Puntaje datacredito bajo (estandarizado, bajo la media = negativo)
            if row["num__puntaje_datacredito"] < self.credit_score_threshold:
                riesgo += 1

            # Muchas consultas
            if row["num__huella_consulta"] > self.consultas_threshold:
                riesgo += 1

            # Independiente
            if row["cat__tipo_laboral_Independiente"] == 1:
                riesgo += 1

            # 2 o más señales de riesgo → no paga a tiempo
            predictions.append(0 if riesgo >= 2 else 1)

        return np.array(predictions)

# =============================================================================
# PIPELINE Y VALIDACIÓN CRUZADA
# =============================================================================

model      = HeuristicModel()
model_pipe = Pipeline(steps=[("model", model)])

# StratifiedKFold respeta el desbalance de clases
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Métrica principal: recall clase 0 (detectar morosos)
scorer_recall = make_scorer(recall_score, pos_label=0)

scoring_metrics = {
    "recall_clase0"     : make_scorer(recall_score, pos_label=0),
    "recall"            : "recall",
    "precision"         : "precision",
    "f1"                : "f1",
    "balanced_accuracy" : "balanced_accuracy"
}

print("\n📊 Validación Cruzada (StratifiedKFold, 10 folds):")
print("="*55)

cv_results = {}
for metric_name, scorer in scoring_metrics.items():
    scores = cross_val_score(
        model_pipe,
        X_train,
        y_train,
        cv=skfold,
        scoring=scorer
    )
    cv_results[metric_name] = scores
    print(f"{metric_name:20s} → mean: {scores.mean():.3f} | std: {scores.std():.3f}")

# =============================================================================
# EVALUACIÓN FINAL EN TEST
# =============================================================================

model_pipe.fit(X_train, y_train)
y_pred = model_pipe.predict(X_test)

print("\n📋 Reporte de Clasificación (Test):")
print("="*55)
print(classification_report(
    y_test,
    y_pred,
    target_names=["No paga (0)", "Paga (1)"]
))

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No paga (0)", "Paga (1)"]
)
plt.title("Matriz de Confusión - Modelo Heurístico")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix_heuristic.png"))
plt.show()

# =============================================================================
# LEARNING CURVE
# =============================================================================

common_params = {
    "X"           : X_train,
    "y"           : y_train,
    "train_sizes" : np.linspace(0.1, 1.0, 5),
    "cv"          : ShuffleSplit(n_splits=20, test_size=0.2, random_state=42),
    "n_jobs"      : -1,
    "return_times": True,
    "scoring"     : scorer_recall
}

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
    model_pipe, **common_params
)

train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
test_mean  = np.mean(test_scores, axis=1)
test_std   = np.std(test_scores, axis=1)

# Gráfica learning curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, "o-", label="Train score")
ax.plot(train_sizes, test_mean,  "o-", color="orange", label="CV score")
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.3)
ax.fill_between(train_sizes, test_mean-test_std,   test_mean+test_std,   alpha=0.3, color="orange")
ax.set_title("Learning Curve - Modelo Heurístico (Recall clase 0)")
ax.set_xlabel("Training examples")
ax.set_ylabel("Recall clase 0")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "learning_curve_heuristic.png"))
plt.show()

# =============================================================================
# ESCALABILIDAD
# =============================================================================

fit_times_mean  = np.mean(fit_times, axis=1)
fit_times_std   = np.std(fit_times, axis=1)
score_times_mean = np.mean(score_times, axis=1)
score_times_std  = np.std(score_times, axis=1)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

ax[0].plot(train_sizes, fit_times_mean, "o-")
ax[0].fill_between(train_sizes, fit_times_mean-fit_times_std,
                   fit_times_mean+fit_times_std, alpha=0.3)
ax[0].set_ylabel("Fit time (s)")
ax[0].set_title("Escalabilidad - Modelo Heurístico")

ax[1].plot(train_sizes, score_times_mean, "o-")
ax[1].fill_between(train_sizes, score_times_mean-score_times_std,
                   score_times_mean+score_times_std, alpha=0.3)
ax[1].set_ylabel("Score time (s)")
ax[1].set_xlabel("Número de muestras de entrenamiento")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "scalability_heuristic.png"))
plt.show()

print("\n Modelo heurístico completado")