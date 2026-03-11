# -*- coding: utf-8 -*-
# =============================================================================
# model_monitoring.py
# Monitoreo del modelo desplegado con detección de Data Drift
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import joblib
from datetime import datetime
from scipy.stats import ks_2samp

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TEST_PATH      = os.path.join(BASE_DIR, "df_test.xlsx")
METRICS_PATH   = os.path.join(BASE_DIR, "metricas_modelo.xlsx")
MODEL_PATH     = os.path.join(BASE_DIR, "mejor_modelo.pkl")
LOG_PATH       = os.path.join(BASE_DIR, "predicciones_log.xlsx")
API_URL        = "http://localhost:8000/predecir/batch"
PERIODICIDAD   = "Mensual"
SAMPLE_SIZE    = 100  # registros a muestrear por período

FEATURES = [
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

# =============================================================================
# FUNCIONES
# =============================================================================

def cargar_datos():
    """Carga el dataset de test y las métricas baseline"""
    df_test    = pd.read_excel(TEST_PATH)
    df_metrics = pd.read_excel(METRICS_PATH)
    modelo     = joblib.load(MODEL_PATH)
    return df_test, df_metrics, modelo


def muestrear_periodo(df, sample_size=SAMPLE_SIZE):
    """Toma una muestra aleatoria del período actual"""
    return df.sample(
        n=min(sample_size, len(df)),
        random_state=int(datetime.now().strftime("%Y%m"))
    )


def enviar_predicciones(df_sample):
    """
    Envía los datos al endpoint de FastAPI y retorna
    las predicciones junto con los datos originales.
    """
    try:
        payload = {"clientes": df_sample[FEATURES].to_dict(orient="records")}
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        df_result = df_sample.copy().reset_index(drop=True)
        df_result["prediccion"]       = [p["prediccion"]       for p in data["predicciones"]]
        df_result["probabilidad_0"]   = [p["probabilidad_0"]   for p in data["predicciones"]]
        df_result["probabilidad_1"]   = [p["probabilidad_1"]   for p in data["predicciones"]]
        df_result["riesgo"]           = [p["riesgo"]           for p in data["predicciones"]]
        df_result["fecha_prediccion"] = datetime.now().strftime("%Y-%m-%d")

        return df_result, data["total_riesgo_alto"], data["total_riesgo_bajo"]

    except Exception as e:
        st.error(f"❌ Error al conectar con la API: {e}")
        return None, 0, 0


def detectar_drift(df_referencia, df_actual, features):
    """
    Detecta Data Drift usando el test de Kolmogorov-Smirnov.
    Compara la distribución de las features entre el dataset
    de referencia (train) y el período actual.
    """
    resultados_drift = []

    for feature in features:
        if feature in df_referencia.columns and feature in df_actual.columns:
            stat, p_value = ks_2samp(
                df_referencia[feature].dropna(),
                df_actual[feature].dropna()
            )
            resultados_drift.append({
                "Feature"   : feature,
                "KS Stat"   : round(stat, 4),
                "P-Value"   : round(p_value, 4),
                "Drift"     : "🔴 Sí" if p_value < 0.05 else "🟢 No"
            })

    return pd.DataFrame(resultados_drift)


def guardar_log(df_result):
    """Guarda las predicciones en un log Excel"""
    if os.path.exists(LOG_PATH):
        df_existing = pd.read_excel(LOG_PATH)
        df_log      = pd.concat([df_existing, df_result], ignore_index=True)
    else:
        df_log = df_result

    df_log.to_excel(LOG_PATH, index=False)


# =============================================================================
# TABLERO STREAMLIT
# =============================================================================

def main():

    st.set_page_config(
        page_title="Monitoreo - Modelo de Créditos",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Monitoreo del Modelo de Predicción de Créditos")
    st.markdown(f"**Periodicidad:** {PERIODICIDAD} | **Muestra por período:** {SAMPLE_SIZE} registros")
    st.markdown("---")

    # Cargar datos
    df_test, df_metrics, modelo = cargar_datos()

    # =========================================================================
    # SECCIÓN 1: MÉTRICAS BASELINE
    # =========================================================================
    st.header("📋 Métricas Baseline del Modelo")

    col1, col2, col3, col4 = st.columns(4)

    metricas_dict = dict(zip(df_metrics["Métrica"], df_metrics["Valor"]))

    col1.metric(
        "Recall clase 0",
        f"{metricas_dict.get('Recall clase 0 (morosos)', 0):.3f}",
        help="Porcentaje de morosos detectados"
    )
    col2.metric(
        "Balanced Accuracy",
        f"{metricas_dict.get('Balanced Accuracy', 0):.3f}",
        help="Accuracy balanceada entre clases"
    )
    col3.metric(
        "F1 clase 0",
        f"{metricas_dict.get('F1 clase 0', 0):.3f}",
        help="F1 score para morosos"
    )
    col4.metric(
        "ROC AUC",
        f"{metricas_dict.get('ROC AUC', 0):.3f}",
        help="Area bajo la curva ROC"
    )

    st.markdown("---")

    # =========================================================================
    # SECCIÓN 2: MUESTREO Y PREDICCIONES
    # =========================================================================
    st.header(f"🔄 Muestreo {PERIODICIDAD}")

    if st.button("▶️ Ejecutar muestreo del período actual"):

        with st.spinner("Tomando muestra y consultando API..."):

            # Muestrear
            df_sample = muestrear_periodo(df_test)

            # Enviar al endpoint
            df_result, total_alto, total_bajo = enviar_predicciones(df_sample)

            if df_result is not None:

                # Guardar log
                guardar_log(df_result)

                # Métricas del período
                st.subheader("📊 Resultados del período")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total clientes", len(df_result))
                col2.metric("🔴 Riesgo Alto", total_alto)
                col3.metric("🟢 Riesgo Bajo", total_bajo)

                # Tabla de predicciones
                st.subheader("📋 Tabla de predicciones")
                st.dataframe(
                    df_result[FEATURES[:5] + ["prediccion", "probabilidad_0", "riesgo"]],
                    use_container_width=True
                )

                # Distribución de predicciones
                st.subheader("📈 Distribución de predicciones")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Pie chart
                axes[0].pie(
                    [total_alto, total_bajo],
                    labels=["Riesgo Alto", "Riesgo Bajo"],
                    colors=["#ff4444", "#44bb44"],
                    autopct="%1.1f%%"
                )
                axes[0].set_title("Distribución de Riesgo")

                # Histograma probabilidades
                axes[1].hist(df_result["probabilidad_0"], bins=20, color="#ff4444", alpha=0.7)
                axes[1].set_title("Distribución Probabilidad clase 0")
                axes[1].set_xlabel("Probabilidad de no pago")
                axes[1].set_ylabel("Frecuencia")

                st.pyplot(fig)
                plt.close()

    st.markdown("---")

    # =========================================================================
    # SECCIÓN 3: DATA DRIFT
    # =========================================================================
    st.header("🔍 Detección de Data Drift")
    st.markdown("Comparación de distribuciones entre datos de referencia y período actual usando **KS Test**")

    if st.button("🔍 Detectar Data Drift"):

        with st.spinner("Analizando distribuciones..."):

            df_referencia = df_test.sample(n=500, random_state=42)
            df_actual     = muestrear_periodo(df_test)

            df_drift = detectar_drift(df_referencia, df_actual, FEATURES)

            # Resumen
            n_drift = (df_drift["Drift"] == "🔴 Sí").sum()
            st.subheader(f"📊 Resultado: {n_drift}/{len(FEATURES)} features con drift")

            if n_drift == 0:
                st.success("✅ No se detectó Data Drift en ninguna feature")
            elif n_drift <= 3:
                st.warning(f"⚠️ Se detectó Data Drift leve en {n_drift} features")
            else:
                st.error(f"🚨 Se detectó Data Drift severo en {n_drift} features — revisar modelo")

            st.dataframe(df_drift, use_container_width=True)

            # Gráfica KS Stats
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = ["#ff4444" if d == "🔴 Sí" else "#44bb44" for d in df_drift["Drift"]]
            ax.barh(df_drift["Feature"], df_drift["KS Stat"], color=colors)
            ax.axvline(x=0.05, color="black", linestyle="--", label="Umbral p=0.05")
            ax.set_title("KS Statistic por Feature")
            ax.set_xlabel("KS Statistic")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # =========================================================================
    # SECCIÓN 4: HISTORIAL DE PREDICCIONES
    # =========================================================================
    st.header("📅 Historial de Predicciones")

    if os.path.exists(LOG_PATH):
        df_log = pd.read_excel(LOG_PATH)
        st.metric("Total predicciones registradas", len(df_log))
        st.dataframe(df_log.tail(50), use_container_width=True)
    else:
        st.info("Aún no hay predicciones registradas. Ejecuta el muestreo para comenzar.")


if __name__ == "__main__":
    main()