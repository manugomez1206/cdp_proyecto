# -*- coding: utf-8 -*-
# =============================================================================
# vertex_pipeline.py
# Pipeline custom con Vertex Pipelines para predicción de créditos
# =============================================================================

import os
from google.cloud import aiplatform
from kfp import dsl, compiler
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_ID    = "cdp-produccion-490821"
REGION        = "us-central1"
BUCKET_NAME   = "cdp-proyecto-datos"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

# =============================================================================
# COMPONENTES DEL PIPELINE
# =============================================================================

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "google-cloud-storage", "scikit-learn",
                         "imbalanced-learn", "joblib", "openpyxl"]
)
def cargar_datos(
    bucket_name: str,
    file_name: str,
    dataset: Output[Dataset]
):
    """Carga los datos desde GCS"""
    from google.cloud import storage
    import pandas as pd
    import io

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(file_name)
    data   = blob.download_as_bytes()
    df     = pd.read_csv(io.BytesIO(data))

    df.to_csv(dataset.path, index=False)
    print(f"Datos cargados: {df.shape}")


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "imbalanced-learn", "joblib"]
)
def preprocesar_datos(
    dataset: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    """Preprocesa y divide los datos"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import numpy as np

    df = pd.read_csv(dataset.path)

    # Eliminar columnas no necesarias
    cols_eliminar = ['fecha_prestamo', 'saldo_mora_codeudor', 'puntaje',
                     'cant_creditosvigentes', 'saldo_total',
                     'promedio_ingresos_datacredito', 'tendencia_ingresos']
    df = df.drop(columns=[c for c in cols_eliminar if c in df.columns])

    # Imputar nulos
    for col in ['puntaje_datacredito', 'saldo_mora', 'saldo_principal']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Nuevas variables
    df['ratio_cuota_salario']  = df['cuota_pactada'] / (df['salario_cliente'] + 1)
    df['salario_log']          = np.log1p(df['salario_cliente'])
    df['capital_prestado_log'] = np.log1p(df['capital_prestado'])
    df['total_creditos']       = df['creditos_sectorFinanciero'] + \
                                 df['creditos_sectorCooperativo'] + \
                                 df['creditos_sectorReal']

    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    train_df = X_train.copy()
    train_df['Pago_atiempo'] = y_train
    test_df  = X_test.copy()
    test_df['Pago_atiempo']  = y_test

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path,   index=False)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "imbalanced-learn", "joblib"]
)
def entrenar_modelo(
    train_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics]
):
    """Entrena el modelo de regresión logística con SMOTE"""
    import pandas as pd
    import joblib
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (recall_score, balanced_accuracy_score, f1_score)
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    df      = pd.read_csv(train_data.path)
    X_train = df.drop(columns=['Pago_atiempo'])
    y_train = df['Pago_atiempo']

    num_features = ['capital_prestado_log', 'plazo_meses', 'edad_cliente',
                    'salario_log', 'ratio_cuota_salario', 'puntaje_datacredito',
                    'huella_consulta', 'total_creditos']
    cat_features = ['tipo_credito', 'tipo_laboral']

    num_features = [f for f in num_features if f in X_train.columns]
    cat_features = [f for f in cat_features if f in X_train.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', SGDClassifier(loss='log_loss', class_weight='balanced',
                                random_state=42, max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_train)

    recall  = recall_score(y_train, y_pred, pos_label=0)
    bal_acc = balanced_accuracy_score(y_train, y_pred)
    f1      = f1_score(y_train, y_pred, pos_label=0)

    metrics.log_metric("recall_clase0",     recall)
    metrics.log_metric("balanced_accuracy", bal_acc)
    metrics.log_metric("f1_clase0",         f1)

    print(f"Recall clase 0: {recall:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"F1 clase 0: {f1:.3f}")

    joblib.dump(pipeline, model.path + ".pkl")


@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "imbalanced-learn", "joblib"]
)
def evaluar_modelo(
    test_data: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics]
):
    """Evalúa el modelo en el conjunto de prueba"""
    import pandas as pd
    import joblib
    from sklearn.metrics import (recall_score, balanced_accuracy_score,
                                  f1_score, roc_auc_score, average_precision_score)

    df     = pd.read_csv(test_data.path)
    X_test = df.drop(columns=['Pago_atiempo'])
    y_test = df['Pago_atiempo']

    pipeline = joblib.load(model.path + ".pkl")
    y_pred   = pipeline.predict(X_test)
    y_prob   = pipeline.predict_proba(X_test)[:, 0]

    recall   = recall_score(y_test, y_pred, pos_label=0)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, pos_label=0)
    roc_auc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob, pos_label=0)

    metrics.log_metric("recall_clase0",       recall)
    metrics.log_metric("balanced_accuracy",   bal_acc)
    metrics.log_metric("f1_clase0",           f1)
    metrics.log_metric("roc_auc",             roc_auc)
    metrics.log_metric("average_precision",   avg_prec)

    print(f"Recall clase 0:    {recall:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"F1 clase 0:        {f1:.3f}")
    print(f"ROC AUC:           {roc_auc:.3f}")
    print(f"Average Precision: {avg_prec:.3f}")


# =============================================================================
# DEFINICIÓN DEL PIPELINE
# =============================================================================

@dsl.pipeline(
    name="cdp-creditos-pipeline",
    description="Pipeline custom para predicción de pago de créditos"
)
def creditos_pipeline(
    bucket_name: str = BUCKET_NAME,
    file_name: str   = "Base_de_datos.csv"
):
    cargar = cargar_datos(
        bucket_name=bucket_name,
        file_name=file_name
    ).set_cpu_limit('1').set_memory_limit('4G')

    preproc = preprocesar_datos(
        dataset=cargar.outputs["dataset"]
    ).set_cpu_limit('1').set_memory_limit('4G')

    entrenar = entrenar_modelo(
        train_data=preproc.outputs["train_data"]
    ).set_cpu_limit('1').set_memory_limit('4G')

    evaluar = evaluar_modelo(
        test_data=preproc.outputs["test_data"],
        model=entrenar.outputs["model"]
    ).set_cpu_limit('1').set_memory_limit('4G')


# =============================================================================
# COMPILAR Y EJECUTAR EL PIPELINE
# =============================================================================

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=creditos_pipeline,
        package_path="creditos_pipeline.yaml"
    )
    print("Pipeline compilado: creditos_pipeline.yaml")

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name   = "cdp-creditos-pipeline",
        template_path  = "creditos_pipeline.yaml",
        pipeline_root  = PIPELINE_ROOT,
        enable_caching = False
    )
    job.submit()
    print(f"Pipeline enviado")
    print(f"Ver en: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={PROJECT_ID}")