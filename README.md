# Detección de Transacciones Fraudulentas con Ataques Coordinados

## Descripción del Proyecto

Este proyecto fue desarrollado para la clase de Security Data Science de la UNIVERSIDAD DEL VALLE DE GUATEMALA.

Su objetivo es identificar transacciones fraudulentas que forman parte de ataques coordinados mediante el uso del algoritmo LightGBM (Light Gradient Boosting Machine). El sistema se enfoca en detectar patrones específicos de fraude para campañas organizadas de ataques cibernéticos.

### Objetivos Principales

- Crear características (features) que capturen comportamientos anómalos indicativos de ataques coordinados.
- Implementar un modelo de machine learning capaz de detectar transacciones fraudulentas coordinadas.
- Desarrollar funciones de evaluación especializadas para diferentes tipos de patrones fraudulentos.
- Evaluar y comparar el rendimiento de las diferentes funciones de evaluación.

## Características del Proyecto

### Preprocesamiento de Datos
- **Limpieza de datos**: Transformación de los datos a formatos que mejoren su análisis.
- **Ingeniería de características**: Creación de variables especializadas para detectar patrones anómalos.
- **Definición de variable objetivo**: Definición de la variable objetivo para detección de transacciones que sean parte de fraudes colectivos.

### Variables Creadas para Detección

El proyecto incluye la creación de múltiples variables especializadas:

1. **Análisis Temporal**:
   - `time_diff_seconds`: Diferencia de tiempo entre transacciones consecutivas
   - `hour_window`: Ventana de tiempo por hora
   - `trans_per_hour`: Transacciones por hora por tarjeta
   - `hour_trans_ratio`: Ratio de transacciones por hora

2. **Análisis Geográfico**:
   - `unusual_distance`: Distancias inusuales (>100km) entre cliente y comerciante
   - `velocity_km_h`: Velocidad requerida entre transacciones consecutivas
   - `dist_z_score`: Desviaciones estándar en distancia respecto al promedio

3. **Análisis de Montos**:
   - `amt_month_ratio`: Ratio de montos mensuales
   - `amt_year_ratio`: Ratio de montos anuales
   - `amt_z_score`: Desviaciones estándar en montos por categoría
   - `high_amt_first_time`: Primera transacción de alto monto en comerciante

4. **Análisis de Patrones de Comerciantes**:
   - `unique_cards_per_hour`: Tarjetas únicas utilizadas por hora en cada comerciante
   - `amt_variance_hour`: Varianza de montos en ventanas de tiempo
   - `times_day_z_score`: Frecuencia anómala de compras por día

### Modelos Implementados

1. **Basic LGBM**: Modelo base usando LightGBM estándar
2. **Unique Cards**: Modelo LightGBM con función de evaluación enfocada en detectar uso excesivo de tarjetas únicas
3. **Merchant Frequency**: Modelo LightGBM con función de evaluación especializada en frecuencias de compra inusuales
4. **Distance Anomaly**: Modelo LightGBM con función de evaluación orientada a detectar anomalías geográficas

### Librerías y Dependencias

El proyecto requiere las siguientes librerías de Python:

```python
# Manipulación y análisis de datos
pandas
numpy

# Visualización
matplotlib
seaborn

# Machine Learning
scikit-learn
lightgbm

# Procesamiento de datos
scipy

# Utilidades de Python
bisect (biblioteca estándar)
time (biblioteca estándar)
itertools (biblioteca estándar)
```

### Instalación de Dependencias

Para instalar todas las dependencias necesarias, ejecute:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
```

## Cómo Ejecutar el Proyecto

### 1. Preparación del Entorno

```bash
# Clonar o descargar el proyecto
cd proyecto_final

# Instalar dependencias
pip install -r requirements.txt  # Si existe, o usar el comando anterior
```

### 2. Ejecución de Notebooks

El proyecto debe ejecutarse en el siguiente orden:

#### Paso 1: Análisis Exploratorio
```bash
jupyter notebook code/analysis.ipynb
```
Este notebook proporciona:
- Exploración del dataset original (1,852,394 transacciones, 35 columnas)
- Análisis de distribuciones de variables numéricas y categóricas
- Identificación del desbalance de clases en la variable objetivo

#### Paso 2: Transformación y Ingeniería de Características
```bash
jupyter notebook code/transformations.ipynb
```
Este notebook incluye:
- Limpieza de datos (conversión a minúsculas, formato de fechas)
- Creación de 20+ características especializadas
- Definición de la variable `is_coordinated_attack`
- Exportación del dataset procesado

#### Paso 3: Modelado y Evaluación
```bash
jupyter notebook code/model.ipynb
```
Este notebook contiene:
- Preparación de datos para entrenamiento
- Implementación de 4 modelos diferentes
- Evaluación con métricas especializadas
- Análisis de resultados y comparación de modelos

### Métricas de Evaluación

Los modelos fueron evaluados usando:
- Matriz de confusión
- Curva ROC-AUC
- Precision, Recall, F1-Score
- Accuracy

## Herramientas y Tecnologías Utilizadas

- **Python 3.7+**: Lenguaje de programación principal
- **Jupyter Notebooks**: Entorno de desarrollo interactivo
- **LightGBM**: Algoritmo principal de gradient boosting
- **Scikit-learn**: Métricas de evaluación y preprocesamiento
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Matplotlib/Seaborn**: Visualización de datos

## Limitaciones y Consideraciones

1. **Desbalance de clases**: Significativa diferencia entre transacciones fraudulentas y legítimas
2. **Falsos positivos**: Gracias al desbalance que ocurre en la cantidad de transacciones fraudulentas, el podelo puede generar una gran cantidad de falsos positivos
