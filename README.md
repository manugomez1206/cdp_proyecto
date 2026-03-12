# CDP - Predicción de Pago de Créditos
**Ciencia de Datos en Producción | Universidad Pontifica Bolivariana**  
**Autora:** Manuela Gómez Gallego

---

## Descripción
Pipeline MLOps para la predicción de pago oportuno de créditos en una entidad financiera colombiana. El modelo predice si un cliente pagará o no a tiempo su crédito, con especial énfasis en la detección de clientes morosos (clase minoritaria).

- **Variable objetivo:** `Pago_atiempo` (1 = paga, 0 = no paga)
- **Desbalance:** 95% paga / 5% no paga
- **Dataset:** 10.763 registros, 23 columnas originales

---

## Estructura del Repositorio
```
cdp_proyecto/
├── .github/
│   └── workflows/
│       └── build.yml          # CI/CD con SonarCloud
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb         # Carga y exploración inicial
│       ├── comprension_eda.ipynb      # Análisis exploratorio
│       ├── ft_engineering.py          # Feature engineering
│       ├── heuristic_model.py         # Modelo heurístico baseline
│       ├── model_training.py          # Entrenamiento de modelos
│       ├── model_evaluation.py        # Evaluación del mejor modelo
│       ├── model_deploy.py            # API REST con FastAPI
│       ├── model_monitoring.py        # Monitoreo con Streamlit
│       ├── config.json                # Configuración del proyecto
│       ├── Base_de_datos.xlsx         # Dataset original
│       ├── df_train.xlsx              # Datos de entrenamiento
│       ├── df_test.xlsx               # Datos de prueba
│       └── mejor_modelo.pkl           # Mejor modelo entrenado
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── sonar-project.properties
├── set_up.bat
└── readme.md
```

---

## ⚙️ Requisitos
- Python 3.11
- pip
- Git

---

## Instalación
1. Clona el repositorio:
```bash
git clone https://github.com/manugomez1206/cdp_proyecto.git
cd cdp_proyecto
```

2. Ejecuta el script de configuración:
```bash
set_up.bat
```

O manualmente:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Uso

### Feature Engineering
```bash
python mlops_pipeline/src/ft_engineering.py
```

### Entrenamiento del modelo
```bash
python mlops_pipeline/src/model_training.py
```

### Evaluación del modelo
```bash
python mlops_pipeline/src/model_evaluation.py
```

### Deploy — API REST
```bash
python mlops_pipeline/src/model_deploy.py
```
API disponible en: http://localhost:8000/docs

### Monitoreo
```bash
streamlit run mlops_pipeline/src/model_monitoring.py
```
Tablero disponible en: http://localhost:8501

---

## Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| Recall clase 0 (morosos) | 0.592 |
| Balanced Accuracy | 0.624 |
| F1 clase 0 | 0.140 |
| ROC AUC | 0.308 |

**Mejor modelo:** SGDClassifier (Regresión Logística con gradiente descendente)  
**Estrategia desbalance:** SMOTE  
**Validación:** StratifiedKFold (10 folds)

---

## Pipeline MLOps
```
Datos → EDA → Feature Engineering → Modelo Heurístico
                                  → Model Training → Evaluación → Deploy → Monitoreo
```

---

## Docker
```bash
docker-compose up
```

---

## Calidad de Código
Configurado con **SonarCloud** para análisis automático de:
- Calidad del código
- Seguridad
- Cobertura
- Integridad y estilo

---

## Flujo Git
```
feature1 → develop → master (v1.0.0)
feature2 → develop → master (v1.1.0)
```