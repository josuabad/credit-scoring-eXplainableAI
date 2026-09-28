# 💳 Credit Risk Scoring & Multiclass Machine Learning Pipeline

Este repositorio contiene un sistema end-to-end de **Credit Risk Scoring** orientado a la clasificación de clientes según su nivel de riesgo crediticio (_Poor_, _Standard_, _Good_).

El objetivo principal de este proyecto es construir un pipeline de Machine Learning robusto, auditable y listo para producción, combinando técnicas avanzadas de **Data Wrangling**, **Feature Engineering sin Data Leakage**, **Algoritmos basados en Gradient Boosting (LightGBM)** y **Explicabilidad de Modelos (SHAP)** para la toma de decisiones en el sector bancario.

---

## 🛠️ Arquitectura del Proyecto

El flujo de trabajo se estructura en **5 fases clave**, representadas secuencialmente en los cuadernos de ejecución e inferencia:

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Fase 1: Data   │ ──> │ Fase 2: Pipelines & │ ──> │   Fase 3: Modelado   │
│   Wrangling     │     │ Feature Engineering │     │     Multiclase       │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
                                                               │
┌─────────────────┐     ┌─────────────────────┐                │
│ Fase 5: Módulo  │ <── │ Fase 4: Capa XAI    │ <──────────────┘
│  de Inferencia  │     │   (SHAP Multiclass) │
└─────────────────┘     └─────────────────────┘
```

1. **Fase 1: Data Wrangling y Limpieza**
   - Eliminación de identificadores irrelevantes (`ID`, `Customer_ID`, `SSN`, etc.).
   - Limpieza mediante **Regex** de caracteres parásito en columnas numéricas (ej. `'19114.12*'`, `'28_'`).
   - Tratamiento de datos atípicos/incoherentes (ej. edades fuera del rango $[18, 100]$ o valores negativos ilógicos) convirtiéndolos a `NaN`.
   - Parsing de la variable `Credit_History_Age` (ej. _"22 Years and 1 Months"_) a meses totales (`Credit_History_Age_Months`).
   - Mapeo ordinal de la variable objetivo: `{Poor: 0, Standard: 1, Good: 2}`.

2. **Fase 2: Ingeniería de Características y Preprocesamiento**
   - Estratificación `train_test_split` (80/20) respetando la distribución multiclase del `Credit_Score`.
   - Encapsulamiento del preprocesamiento en un `ColumnTransformer` de Scikit-Learn para prevenir **Data Leakage**:
     - **Variables Numéricas**: Imputación por mediana + `StandardScaler`.
     - **Variables Categóricas**: Imputación por categoría constante (`Unknown`) + `OneHotEncoder(handle_unknown='ignore')`.

3. **Fase 3: Modelado Multiclase**
   - Evaluación comparativa entre un modelo base (**Logistic Regression**) y un modelo avanzado de Gradient Boosting (**LightGBM**).
   - Métricas enfocas en negocio: **F1-Score Macro** e inspección de errores críticos en la **Matriz de Confusión** (reducir falsos positivos donde clientes _Poor/Riesgo Alto_ sean clasificados como _Good/Riesgo Bajo_).

4. **Fase 4: Capa de Explicabilidad (XAI con SHAP)**
   - Extracción dinámica de nombres de características tras la codificación _One-Hot_.
   - **Explicación Global**: `shap.summary_plot` / beeswarm para identificar las variables con mayor peso estratégico (ej. `Delay_from_due_date`, `Outstanding_Debt`).
   - **Explicación Local**: Gráficos en cascada (`shap.plots.waterfall`) para dar soporte al _"Derecho a la Explicación"_ del cliente en auditorías bancarias.

5. **Fase 5: Módulo y Pipeline de Inferencia**
   - Persistencia de los objetos encapsulados (`Pipeline` + `ColumnTransformer` + `Model`) en formato `.joblib`.
   - Construcción de `inference.py` con una función end-to-end (`predict_credit_risk`) capaz de transformar un payload JSON/diccionario crudo con ruido en una predicción estructurada con probabilidades por clase.

---

## 📊 Resultados y Evaluación

| Modelo                             | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) | Accuracy |
| :--------------------------------- | :-------------------: | :----------------: | :------------------: | :------: |
| **Logistic Regression (Baseline)** |         0.73          |        0.72        |         0.71         |   0.74   |
| **LightGBM (Modelo Campeón)**      |       **0.71**        |      **0.72**      |       **0.71**       | **0.73** |

> **Nota Financiera:** El valor principal del modelo Gradient Boosting no solo recae en el rendimiento global, sino en su capacidad no lineal para reducir sustancialmente las falsas aprobaciones en la clase de mayor riesgo (_Poor_).

---

## 📂 Estructura de Directorios

```text
.
│   .gitignore
│   docker-compose.yml
│   README.md
│   requirements.txt
│   LICENSE
│
└───workspace
    ├───data
    │   ├───models
    │   └───raw
    │
    └───notebooks
        │   Fase1.ipynb
        │   Fase2-3-4.ipynb
        │   Fase5.ipynb

```

---

## 🚀 Uso del Módulo de Inferencia

Para evaluar una nueva solicitud de crédito en bruto, se puede importar la función `predict_credit_risk`:

```python
from src.inference import predict_credit_risk

# Payload crudo de un cliente nuevo
nuevo_cliente = {
    "Age": "32",
    "Occupation": "Scientist",
    "Annual_Income": "12500.50_",
    "Monthly_Inhand_Salary": 1041.70,
    "Num_Bank_Accounts": 6,
    "Num_Credit_Card": 8,
    "Interest_Rate": 25,
    "Num_of_Loan": 9,
    "Type_of_Loan": "Payday Loan, Personal Loan",
    "Delay_from_due_date": 28,
    "Num_of_Delayed_Payment": "18",
    "Changed_Credit_Limit": 12.5,
    "Num_Credit_Inquiries": 12,
    "Credit_Mix": "Bad",
    "Outstanding_Debt": "4500.80_",
    "Credit_Utilization_Ratio": 38.5,
    "Credit_History_Age": "2 Years and 3 Months",
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 150.0,
    "Amount_invested_monthly": "50.0_",
    "Payment_Behaviour": "High_spent_Small_value_payments",
    "Monthly_Balance": 120.0
}

# Ejecución de la inferencia
resultado = predict_credit_risk(nuevo_cliente)
print(resultado)
```

**Respuesta devuelta:**

```json
{
  "score_predicho": "Standard (Riesgo Medio)",
  "probabilidades": {
    "Poor": 30.76,
    "Standard": 64.97,
    "Good": 4.27
  }
}
```

---

## ⚙️ Instalación y Requisitos

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/josuabad/credit-scoring-eXplainableAI.git
   cd credit-scoring-eXplainableAI
   ```

2. **Crear y activar un entorno virtual:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

_Se puede usar un entorno virtual para aislar las dependencias del proyecto o usar el archivo docker-compose y desplegar el entorno en un contenedor._

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

### Requisitos Principales

- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `shap`
- `joblib`
- `matplotlib`

**Data source**: [Kaggle: Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
