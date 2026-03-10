# -*- coding: utf-8 -*-
# =============================================================================
# ft_engineering.py
# Pipeline de Feature Engineering para modelo de créditos
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# =============================================================================
# TRANSFORMADORES CUSTOM
# =============================================================================

class EliminarColumnas(BaseEstimator, TransformerMixin):
    """
    Elimina columnas no predictivas identificadas en el EDA:
      - fecha_prestamo      : no predictiva directamente
      - saldo_mora_codeudor : 5.5% nulos, +99% ceros, sin aporte informativo
    """
    COLUMNAS_ELIMINAR = [
        "fecha_prestamo",
        "saldo_mora_codeudor",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_presentes = [c for c in self.COLUMNAS_ELIMINAR if c in X.columns]
        return X.drop(columns=cols_presentes).copy()


class Imputacion(BaseEstimator, TransformerMixin):
    """
    Imputa medianas calculadas en fit sobre train para variables con nulos:
      - puntaje_datacredito : 6 nulos
      - saldo_mora          : 156 nulos
      - saldo_principal     : 405 nulos
    """

    def fit(self, X, y=None):
        self.median_puntaje_         = X["puntaje_datacredito"].median()
        self.median_saldo_mora_      = X["saldo_mora"].median()
        self.median_saldo_principal_ = X["saldo_principal"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["puntaje_datacredito"] = X["puntaje_datacredito"].fillna(self.median_puntaje_)
        X["saldo_mora"]          = X["saldo_mora"].fillna(self.median_saldo_mora_)
        X["saldo_principal"]     = X["saldo_principal"].fillna(self.median_saldo_principal_)
        return X


class Outliers(BaseEstimator, TransformerMixin):
    """
    Filtra registros con valores fuera de rangos válidos para el contexto
    colombiano, identificados en el EDA:
      - edad_cliente              : [18, 80] años
      - salario_cliente           : [500_000, 100_000_000] COP
      - total_otros_prestamos     : <= 2_000_000_000 COP
      - puntaje_datacredito       : [150, 999]
      - creditos_sectorFinanciero : <= 20
      - creditos_sectorReal       : <= 15
    IMPORTANTE: Solo se aplica en entrenamiento, no en producción.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = (
            X["edad_cliente"].between(18, 80) &
            X["salario_cliente"].between(500_000, 100_000_000) &
            (X["total_otros_prestamos"] <= 2_000_000_000) &
            X["puntaje_datacredito"].between(150, 999) &
            (X["creditos_sectorFinanciero"] <= 20) &
            (X["creditos_sectorReal"] <= 15)
        )
        return X[mask].copy()


class NuevasVariables(BaseEstimator, TransformerMixin):
    """
    Crea features derivadas con lógica de negocio financiero:
      - ratio_cuota_salario  : % del salario mensual destinado a la cuota
      - salario_log          : log1p del salario (reduce asimetría)
      - capital_prestado_log : log1p del capital (reduce asimetría)
      - total_creditos       : suma de créditos en todos los sectores
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["ratio_cuota_salario"]  = X["cuota_pactada"] / (X["salario_cliente"] + 1)
        X["salario_log"]          = np.log1p(X["salario_cliente"])
        X["capital_prestado_log"] = np.log1p(X["capital_prestado"])
        X["total_creditos"]       = (
            X["creditos_sectorFinanciero"] +
            X["creditos_sectorCooperativo"] +
            X["creditos_sectorReal"]
        )
        return X


class ToDF(BaseEstimator, TransformerMixin):
    """
    Aplica StandardScaler a numéricas y OneHotEncoder a categóricas,
    retornando un DataFrame con nombres de columnas.
    """

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features     = numeric_features
        self.categorical_features = categorical_features
        self.ct_                  = None

    def fit(self, X, y=None):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", ohe,             self.categorical_features),
            ],
            verbose_feature_names_out=True
        )
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        Xt         = self.ct_.transform(X)
        feat_names = self.ct_.get_feature_names_out()
        return pd.DataFrame(Xt, columns=feat_names, index=X.index)


# =============================================================================
# FEATURES
# =============================================================================

numeric_features = [
    "capital_prestado_log",
    "plazo_meses",
    "edad_cliente",
    "salario_log",
    "ratio_cuota_salario",
    "puntaje_datacredito",
    "huella_consulta",
    "total_creditos"
]

categorical_features = [
    "tipo_credito",
    "tipo_laboral"
]

# =============================================================================
# PIPELINE
# =============================================================================

pipeline_features = Pipeline(steps=[
    ("eliminar_columnas", EliminarColumnas()),
    ("imputacion",        Imputacion()),
    ("outliers",          Outliers()),
    ("nuevas_variables",  NuevasVariables()),
    ("to_df",             ToDF(numeric_features, categorical_features))
])

# =============================================================================
# CARGA DE DATOS Y EJECUCIÓN
# =============================================================================

if __name__ == "__main__":

    # Ruta al archivo
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "df_model.xlsx")

    print(f"Cargando datos desde: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    # Separar X e y
    X = df.drop("Pago_atiempo", axis=1)
    y = df["Pago_atiempo"]

    # Train/Test split ANTES del pipeline para evitar data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # Fit solo con train, transform a ambos
    pipeline_features.fit(X_train, y_train)
    X_train_tf = pipeline_features.transform(X_train)
    X_test_tf  = pipeline_features.transform(X_test)

    # Alinear y con el índice resultante después de Outliers
    y_train = y_train.loc[X_train_tf.index]
    y_test  = y_test.loc[X_test_tf.index]

    print(f"✅ Pipeline ejecutado correctamente")
    print(f"📊 Train: {X_train_tf.shape} | Test: {X_test_tf.shape}")
    print(f"🎯 Desbalance train - clase 0: {(y_train==0).sum()} | clase 1: {(y_train==1).sum()}")
    print(f"\nColumnas generadas:\n{list(X_train_tf.columns)}")

    # Guardar datasets procesados
    df_train = X_train_tf.copy()
    df_train["Pago_atiempo"] = y_train.values
    df_train.to_excel(os.path.join(BASE_DIR, "df_train.xlsx"), index=False)

    df_test = X_test_tf.copy()
    df_test["Pago_atiempo"] = y_test.values
    df_test.to_excel(os.path.join(BASE_DIR, "df_test.xlsx"), index=False)

    print(f"\n✅ Archivos guardados:")
    print(f"   → df_train.xlsx ({df_train.shape})")
    print(f"   → df_test.xlsx  ({df_test.shape})")