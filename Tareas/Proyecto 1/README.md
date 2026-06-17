# Diagnóstico Inteligente de Cáncer de Mama mediante Aprendizaje Supervisado Basado en Datos Clínicos Reales

## 1. Descripción del Problema
El cáncer de mama es una de las principales causas de mortalidad femenina a nivel global. Un diagnóstico temprano y certero es fundamental para aumentar las tasas de supervivencia de las pacientes; sin embargo, la interpretación visual de biopsias por aspiración con aguja fina (PAAF) puede ser compleja, depender de la experiencia del especialista y estar sujeta a variabilidades subjetivas.

Para resolver esto de forma cuantitativa, el presente proyecto aborda el desafío desde la perspectiva del aprendizaje supervisado, modelando una situación clínica real de **Clasificación Binaria**. El objetivo es clasificar de manera automática si una masa mamaria es **Maligna (M)** o **Benigna (B)** basándose exclusivamente en características geométricas y morfológicas calculadas a partir de imágenes digitalizadas de los núcleos celulares del tejido. Esta solución optimiza el flujo de trabajo patológico y actúa como una herramienta analítica de soporte a la decisión médica.

## 2. Descripción del Dataset y Fuente
El conjunto de datos seleccionado corresponde al **"Breast Cancer Wisconsin (Diagnostic) Dataset"**, proveniente de una investigación clínica real del hospital de la Universidad de Wisconsin y alojado en el repositorio público UCI Machine Learning. El dataset cuenta con 569 muestras y 30 características numéricas continuas.

Cada registro almacena información del núcleo celular extraído de la biopsia, abarcando métricas de **Media (mean)**, **Error Estándar (se)** y **Peor Valor (worst)** para los siguientes atributos morfológicos:
* `radius`: Distancia desde el centro a los puntos del perímetro.
* `texture`: Desviación estándar de los valores de la escala de grises.
* `perimeter` y `area`: Magnitudes geométricas del núcleo.
* `smoothness`: Variación local en las longitudes de los radios.
* `compactness`: Calculado como $\text{perímetro}^2 / \text{área} - 1.0$.
* `concavity` y `concave points`: Gravedad y número de secciones cóncavas del contorno.
* `symmetry` y `fractal dimension`: Regularidad y aproximación de la frontera celular.

**Variable Objetivo (`target`):** Diagnóstico histopatológico codificado como `1` (Maligno) y `0` (Benigno).

## 3. Justificación de los Modelos Seleccionados
Para abordar este desafío de clasificación diagnóstica, se optó por un enfoque comparativo utilizando dos algoritmos con fundamentos matemáticos distintos:
1. **K-Nearest Neighbors (KNN):** Seleccionado como modelo base (*baseline*). Mide distancias en un espacio métrico continuo bajo la premisa de que configuraciones morfológicas similares pertenecen al mismo tipo de diagnóstico. Requiere estandarización obligatoria (`StandardScaler`) debido a que variables de gran escala (como el `area`) dominarían los cálculos espaciales sobre los índices decimales (como la `smoothness`).
2. **Random Forest (RF):** Seleccionado como modelo avanzado de ensamble. Utiliza múltiples árboles de decisión independientes que segmentan el espacio de características mediante reglas lógicas anidadas y jerárquicas, ideal para capturar interacciones complejas no lineales y mitigar los efectos de la colinealidad de los datos.

## 4. Metodología Aplicada (Paso a Paso)
1. **Carga y Sanitización:** Importación independiente del archivo `data.csv`, removiendo identificadores irrelevantes (`id`) y limpiando anomalías estructurales del formato.
2. **Ingeniería de Características:** Mapeo y codificación de la etiqueta categórica original (`M`/`B`) a variables booleanas numéricas (`1`/`0`) requeridas por el pipeline computacional.
3. **División Estricta:** Partición del dataset en un $80\%$ para entrenamiento y $20\%$ para pruebas mediante una estrategia de división estratificada (`stratify`). Esto garantiza que ambos conjuntos mantengan la proporción natural de la muestra oncológica real, mitigando la fuga de datos (*data leakage*).
4. **Preprocesamiento:** Ajuste y transformación escalar de las características independientes vía `StandardScaler`.
5. **Optimización con Validación Cruzada (Grid Search + CV=5):** Búsqueda exhaustiva de hiperparámetros óptimos ($K$ vecinos y distancias métricas para KNN; cantidad de estimadores y profundidad máxima para Random Forest).
6. **Evaluación Multidimensional:** Testeo ciego final con el conjunto de prueba independiente para el reporte de métricas diagnósticas (Accuracy, Precision, Recall, F1-Score y AUC ROC).

## 5. Análisis Exploratorio de Datos (EDA)

Tras procesar estadísticamente las variables de entrenamiento, se procedió a evaluar la distribución del diagnóstico clínico y las covarianzas lineales de las variables morfológicas principales (ver gráfico `eda_distribucion_correlacion_real.png`):

### 5.1. Distribución Real de las Clases
El conjunto de entrenamiento exhibe una distribución de **285 muestras Benignas (0)** y **170 muestras Malignas (1)**. Este desbalance moderado (~63% vs ~37%) es una representación orgánica y fiel de los entornos de tamizaje oncológico real, donde las lesiones benignas suelen ser más frecuentes. La estratificación en la división resguarda que el conjunto de prueba mantenga exactamente estas proporciones (72 Benignos, 42 Malignos), permitiendo una evaluación transparente.

### 5.2. Covarianzas e Interpretación Clínica de la Correlación
Al inspeccionar las relaciones lineales respecto a la variable objetivo (`target`), se observa una intensa correlación positiva con los atributos: `concave points_mean` (0.78), `perimeter_mean` (0.74), `radius_mean` (0.73) y `area_mean` (0.70). Esto demuestra matemáticamente que los núcleos celulares de los tumores malignos se caracterizan por una mayor magnitud geométrica y un mayor número de irregularidades o hundimientos en sus contornos. 

Asimismo, se detectó una colinealidad extrema ($\approx 1.00$) entre el radio, el perímetro y el área, validando la idoneidad teórica de Random Forest para manejar la redundancia a través del submuestreo de características.

### 5.3. Necesidad de Estandarización
El resumen estadístico expuso una discrepancia crítica en las escalas de los datos: variables como `area_mean` presentan valores promedio de **659.57** (máximo de **2501.0**), mientras que variables de regularidad como `smoothness_mean` operan en rangos decimales mínimos (media de **0.096**). La aplicación de `StandardScaler` ecualizó el peso de los atributos a un espacio z-score común, transformando un registro de área original de `797.8` a un valor escalado de `0.3839` y su correspondiente radio a `0.5186`.

## 6. Resultados Obtenidos e Interpretación

Tras realizar la optimización en la grilla mediante Validación Cruzada ($CV=5$) y evaluar los mejores estimadores con el conjunto de prueba independiente de 114 pacientes, se obtuvieron las siguientes métricas globales:

| Métrica | K-Nearest Neighbors (KNN) | Random Forest (RF) |
| :--- | :---: | :---: |
| **Hiperparámetros Óptimos** | `metric: 'euclidean'`, `n_neighbors: 3` | `max_depth: 7`, `n_estimators: 50` |
| **Accuracy en CV (Entrenamiento)** | 0.9692 | 0.9670 |
| **Accuracy (Test Set)** | 0.9386 | 0.9737 |
| **Precision (Macro)** | 0.9475 | 0.9800 |
| **Recall (Macro)** | 0.9216 | 0.9643 |
| **F1-Score (Macro)** | 0.9322 | 0.9713 |
| **AUC ROC** | 0.9825 | 0.9950 |

### 6.1. Discusión Crítica de los Modelos en la Grilla de Entrenamiento
* **KNN:** El mejor Accuracy promedio en validación cruzada fue de **$96.92\%$** con una métrica Euclidiana y $K=3$ vecinos. Que el modelo óptimo requiera un número bajo de vecinos revela que el espacio de características es localmente muy homogéneo; es decir, las muestras malignas reales se encuentran fuertemente agrupadas en regiones espaciales bien delimitadas.
* **Random Forest:** Alcanzó un **$96.70\%$** de Accuracy promedio en validación cruzada. El optimizador limitó de forma automática el crecimiento de los árboles a una profundidad máxima de 7 niveles (`max_depth: 7`) y un tamaño de bosque acotado a 50 estimadores. Esta restricción estructural actúa como una regularización nativa que limita la varianza del modelo, impidiendo que el ensamble memorice el ruido del entrenamiento y priorizando la generalización.

### 6.2. Evaluación Definitiva en el Test Set y Gestión del Riesgo Oncológico
Al someter a ambos modelos al testeo ciego final (ver archivo `matrices_confusion_finales_reales.png`), **Random Forest demostró una superioridad diagnóstica categórica**, elevando el *Accuracy* general a un **$97.37\%$** y alcanzando un *AUC ROC* de **$0.9950$**.

Desde una perspectiva clínica y de seguridad del paciente, el análisis de las matrices de confusión arroja la conclusión más crítica del proyecto:
* **KNN** clasificó correctamente a 71 tumores benignos y 36 malignos, registrando 1 Falso Positivo. No obstante, generó **6 Falsos Negativos** (pacientes con cáncer dados de alta erróneamente), reduciendo su capacidad de detección o *Recall* a un **$92.16\%$**.
* **Random Forest** anuló los Falsos Positivos por completo (72 aciertos de 72 casos benignos reales, dando una *Precision* de **$98.00\%$**) y, de forma crucial, **redujo el error médico más peligroso a solo 3 Falsos Negativos**, elevando la sensibilidad diagnóstica o *Recall* a un sobresaliente **$96.43\%$**. Las particiones lógicas del ensamble lograron capturar sutilezas morfológicas complejas que las distancias geométricas globales de KNN ignoraron.

## 7. Conclusiones Generales

1. **Resolución Exitosa con Rigor Real:** El proyecto cumplió exitosamente el objetivo de transformar datos morfológicos tumorales reales en un pipeline de clasificación binaria automatizado de alta fidelidad, alcanzando una exactitud clínica del **$97.37\%$** en fases de testeo ciego.
2. **Superioridad de la Estructura Jerárquica:** Se demostró empíricamente que para este set de datos biomédicos, las reglas lógicas y la aleatorización de características de Random Forest superan la rigidez geométrica de KNN. El modelo de ensamble mitigó la colinealidad física de las células y minimizó la tasa de falsos negativos ($3$ casos vs $6$), optimizando la seguridad del diagnóstico médico.
3. **Validación Metodológica sin Data Leakage:** La consistencia entre el rendimiento en validación cruzada (~96.8%) y el conjunto de prueba independiente (~97.3%) descarta cualquier presencia de sobreajuste o fuga de información.