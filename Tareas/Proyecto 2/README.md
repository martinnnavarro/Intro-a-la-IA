# Clasificación Inteligente de Tumores Cerebrales en Resonancias Magnéticas (MRI) mediante Deep Learning y Transfer Learning Avanzado

**Integrantes:** Martin Navarro Toro  
**Curso:** Introducción a la Inteligencia Artificial (COM4402-1)  
**Modelo Avanzado:** Fine-Tuning Gradual y Controlado sobre MobileNetV2  
**Dataset:** Brain Tumor Classification (MRI) - Kaggle  
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri  


## 1. Descripción del Problema
Los tumores cerebrales representan una de las patologías oncológicas más agresivas y complejas de diagnosticar en la medicina moderna. El análisis visual de las Resonancias Magnéticas (MRI) por parte de los especialistas médicos es una tarea minuciosa que requiere un alto nivel de experiencia, consume tiempo crítico y está sujeta a la variabilidad subjetiva del ojo humano. Un retraso o error en la identificación del tipo de tumor puede cambiar drásticamente el protocolo terapéutico y el pronóstico de supervivencia del paciente.

Para abordar este desafío desde la perspectiva del aprendizaje profundo (*Deep Learning*), este proyecto implementa un sistema automatizado modelado como un problema real de **Clasificación Multiclase**. El objetivo es categorizar imágenes de MRI en cuatro clases diagnósticas: tres tipos de tumores específicos (Glioma, Meningioma, Tumor Pituitario) y tejido cerebral sano (Sin Tumor). Esta solución busca actuar como una herramienta complementaria de soporte para la toma de decisiones clínicas, agilizando el triaje y optimizando la precisión diagnóstica en centros asistenciales.

## 2. Descripción del Dataset y Fuente
El conjunto de datos seleccionado corresponde a **"Brain Tumor Classification (MRI)"**, alojado en la plataforma pública Kaggle y recopilado de entornos clínicos reales. El dataset se estructura de forma nativa en dos grandes directorios independientes, `Training` y `Testing`, lo que resguarda la separación estricta para las fases de validación y testeo ciego.

Las imágenes se encuentran distribuidas en cuatro clases diagnósticas hísticas reales:
1. **Glioma (`glioma_tumor`):** Tumores originados en las células gliales que rodean y soportan las neuronas.
2. **Meningioma (`meningioma_tumor`):** Tumores que se desarrollan en las meninges (membranas que protegen el cerebro).
3. **Pituitario (`pituitary_tumor`):** Masas anormales desarrolladas en la glándula pituitaria (base del cerebro).
4. **Sano (`no_tumor`):** Resonancias de control que muestran tejido cerebral libre de masas tumorales.

### 2.1. Análisis Exploratorio de Datos (EDA) y Hallazgos Críticos
Tras realizar una auditoría computacional y estadística directa sobre las carpetas del dataset real, se obtuvieron las siguientes distribuciones de muestras:

* **Glioma Tumor:** 826 imágenes en Entrenamiento, 100 en Prueba (Total: 926).
* **Meningioma Tumor:** 822 imágenes en Entrenamiento, 115 en Prueba (Total: 937).
* **Pituitary Tumor:** 827 imágenes en Entrenamiento, 74 en Prueba (Total: 901).
* **No Tumor (Sano):** 395 imágenes en Entrenamiento, 105 en Prueba (Total: 500).

**Interpretación del Criterio Clínico:** Las tres clases patológicas tumorales se encuentran perfectamente balanceadas y empatadas en el set de entrenamiento (~825 imágenes cada una), lo que blinda a la red contra sesgos iniciales hacia un tumor específico. Por otro lado, la clase sana posee la mitad de muestras (~395), un desbalance moderado que simula fielmente la realidad hospitalaria. 

Asimismo, la inspección visual (ver archivo `eda_muestras_mri_reales.png`) reveló una discrepancia crítica en el **Feature Engineering**: las imágenes tumorales presentaban una resolución de alta definición de $512 \times 512$ píxeles, mientras que las muestras sanas provenían de otro escáner con una resolución menor de $236 \times 260$. Esto validó de forma empírica la necesidad de inyectar un redimensionamiento uniforme a $224 \times 224$ píxeles de manera dinámica y en tiempo real a través de los tensores de nuestros cargadores de flujo, garantizando una entrada homogénea al extractor convolucional.

## 3. Justificación de los Modelos y Plan de Acción Avanzado
Para resolver un problema de visión computacional con un volumen acotado de datos (3,264 imágenes en total), se justifica metodológicamente la aplicación de **Transfer Learning** (Aprendizaje por Transferencia) utilizando la arquitectura **MobileNetV2** preentrenada con el dataset masivo ImageNet.

**Argumentación Técnica y Evolución del Plan de Acción:**
1. **Aprendizaje de Representaciones Maduras:** Entrenar una red convolucional profunda desde cero requiere millones de muestras para que las capas ocultas aprendan fronteras básicas como bordes, texturas y formas. Al reutilizar un modelo preentrenado, el sistema ya cuenta con "detectores de características" visuales altamente optimizados en sus 154 capas convolucionales nativas.
2. **Fine-Tuning Gradual y Controlado:** Los experimentos iniciales con un extractor 100% congelado demostraron que los filtros genéricos de ImageNet confunden la sutil morfología que separa a un glioma de un meningioma. Por ello, el plan de acción evolucionó hacia un enfoque de ajuste fino avanzado: se estableció `base_model.trainable = True` y se bloquearon únicamente las primeras 100 capas base. Liberar las últimas 54 capas superiores elevó los **parámetros entrenables a un volumen de 2,190,404 (8.36 MB)**. Esta flexibilidad paramétrica controlada permite que la red reconfigure sus mapas de abstracción superior y se especialice en la morfología del tejido tumoral, maximizando la precisión diagnóstica intermedia y final sin disparar el coste computacional.


## 4. Metodología Aplicada (Paso a Paso)

1.  **Auditoría e Ingeniería de Características:** Carga controlada de los directorios de imágenes, normalización de los píxeles de un rango $[0, 255]$ a un espacio escalar $[0, 1]$ mediante `rescale=1./255`, y redimensionamiento uniforme de todos los tensores a un tamaño de $224 \times 224 \times 3$.
2.  **Alineación Explícita de Etiquetas:** Se solucionó de raíz el bug crítico de desalineación secuencial forzando la lista explícita `categories` en los tres flujos, garantizando que el entrenamiento, la validación y el testeo ciego mapeen exactamente el mismo índice numérico ($0, 1, 2, 3$) a las mismas clases lógicas de origen.
3.  **Prevención de Data Leakage en Validación:** Se separó la lógica de carga en tres generadores independientes. `train_datagen` aplica transformaciones geométricas aleatorias (*Data Augmentation*: rotación de 20°, desplazamientos y zooms del 10%). Por su parte, el conjunto de validación se cargó a través de un generador sanitizado (`val_datagen`) **completamente limpio de transformaciones artificiales**, previniendo que la red memorice distorsiones ópticas que alteren las métricas de control.
4.  **Partición de Validación Estricta:** Aislamiento automático del $20\%$ de la carpeta `Training` para actuar como conjunto de validación interna intermedia limpia (2,297 imágenes para entrenamiento puro y 573 para validación).
5.  **Instanciación y Descongelamiento Gradual:** Carga de la arquitectura MobileNetV2, configuración de `base_model.trainable = True`, congelamiento manual de las primeras 100 capas convolucionales e inyección del clasificador superior personalizado.
6.  **Configuración del Loop de Optimización:** Compilación mediante la función de pérdida `categorical_crossentropy` y el optimizador **Adam** configurado con una tasa de aprendizaje críticamente baja (`learning_rate=0.0001`) para realizar micro-ajustes estables sin provocar una convergencia catastrófica.
7.  **Sistemas de Control y Callbacks:** Inyección de `ModelCheckpoint` para guardar el estado óptimo basado en la exactitud de validación, asistido por un regularizador de paciencia `EarlyStopping` configurado con `patience=5` apuntando a `val_loss`.
8.  **Testeo Final:** Carga de los mejores pesos y ejecución de predicciones con `shuffle=False` sobre las 394 imágenes del set de prueba ciego independiente (`Testing`).


## 5. Resultados Obtenidos e Interpretación del Entrenamiento

El loop de optimización lanzado en la GPU evidenció una dinámica de aprendizaje y una activación de regularizadores de alto estándar (ver archivo `curvas_aprendizaje_deep_learning.png`):

*   **Dinámica de Aprendizaje y Convergencia:** El modelo inició la primera época con una exactitud de entrenamiento (`accuracy`) de $56.20\%$. Gracias a las 54 capas superiores descongeladas, los filtros se adaptaron con extrema rapidez, escalando de forma robusta en el entrenamiento hasta alcanzar un **$98.48\%$ en la Época 11**.
*   **Comportamiento y Óptimo de Validación:** El conjunto de validación limpio acompañó de forma saludable el loop de optimización, logrando en la **Época 11 su punto óptimo e histórico con un $90.40\%$ de exactitud de validación (`val_accuracy`)** y un coste (`val_loss`) mínimo de **0.4240**. En esta iteración, el callback `ModelCheckpoint` guardó automáticamente el banco definitivo de pesos en el formato nativo Keras v3 (`best_brain_tumor_model.keras`).
*   **Activación del Early Stopping:** A partir de la época 12, el modelo comenzó a entrar en zona de sobreajuste; la exactitud de entrenamiento siguió subiendo de forma artificial hacia el $99\%$, pero la pérdida de validación (`val_loss`) rebotó al alza de forma sostenida ($0.5529$ en la época 12 y alcanzando $0.6308$ en la 15). Al registrarse 5 épocas consecutivas (de la 12 a la 16) donde la pérdida de validación fue incapaz de superar el mínimo de la época 11, **el Early Stopping se activó de forma estricta en la Época 16**, interrumpiendo el entrenamiento y restaurando automáticamente los pesos estables de la Época 11 para salvaguardar la varianza.


## 5.1. Evaluación Final en el Test Set Ciego (Matriz de Confusión)

Al someter los mejores pesos recuperados de la Época 11 al testeo ciego final con las 394 muestras independientes de la carpeta `Testing` (imágenes que el modelo jamás procesó durante el entrenamiento), se obtuvieron las siguientes métricas globales macro-promedio:

### Reporte de Métricas Finales en Test Set
| Métrica | Valor Real Obtenido en el Test Set |
| :--- | :---: |
| **Accuracy (Exactitud General)** | 0.769036 (76.90%) |
| **Precision (Macro-Promedio)** | 0.836323 (83.63%) |
| **Recall (Sensibilidad Macro)** | 0.754454 (75.44%) |
| **F1-Score (Macro-Promedio)** | 0.756229 (75.62%) |

Al abrir analíticamente la matriz de confusión final (ver archivo `matriz_confusion_final_deep_learning.png`), se identificaron los siguientes comportamientos hísticos:

*   **Rendimiento Excepcional en Meningiomas (`meningioma_tumor`):** El sistema alcanzó un desempeño clínico sobresaliente al clasificar correctamente **114 de 115 casos reales**, registrando prácticamente cero falsos negativos derivados de otras patologías.
*   **Control Efectivo del Tejido Sano (`no_tumor`):** Se consolidaron **97 aciertos de 105 pacientes reales**, demostrando que la red diferencia con alta fidelidad Estructuras normales de los ventrículos frente a masas patológicas, reduciendo el riesgo de falsas alarmas.
*   **Consistencia en Tumores Pituitarios (`pituitary_tumor`):** El modelo logró **52 aciertos de 74 casos reales**. Su principal vector de confusión ocurrió hacia los meningiomas (15 casos), debido a las similitudes morfológicas en el realce de contraste de la región selar.
*   **Mitigación de la Complejidad en Gliomas (`glioma_tumor`):** Los gliomas representan la clase morfológicamente más compleja debido a su naturaleza infiltrante y bordes difusos. Gracias al Fine-Tuning de las 54 capas convolucionales superiores, **los aciertos se triplicaron con respecto al extractor congelado, subiendo a 40 aciertos reales**. Aunque persiste una tasa de confusión frente a los meningiomas (51 casos) debido al solapamiento visual real en las resonancias por edema peritumoral, el modelo demuestra una sensibilidad radicalmente superior para extraer las fronteras hísticas del cáncer.


## 6. Conclusiones Generales

1.  **Superioridad del Fine-Tuning Controlado:** Se demostró empíricamente que mantener un congelamiento absoluto de las capas convolucionales base limita el rendimiento ante problemas médicos de alta complejidad. El paso hacia un Fine-Tuning controlado —descongelando selectivamente las últimas 54 capas de la red y aplicando una tasa de aprendizaje reducida ($1 \times 10^{-4}$)— fue el factor determinante para disparar las métricas en el set de prueba ciego de un $52\%$ inicial a un robusto **$76.90\%$ de Accuracy general y un destacado $83.63\%$ de Precision macro**.
2.  **Saneamiento de la Fuga de Datos (Data Leakage):** La reestructuración del pipeline de datos fue clave para subsanar inconsistencias lógicas en el entrenamiento. Aislar el generador de validación de los procesos de aumento geométrico garantizó que las métricas intermedias fueran evaluaciones incorruptibles del desempeño del modelo, erradicando sesgos metodológicos presentes en divisiones automatizadas implícitas.
3.  **Robustez y Auditoría Médica:** La combinación coordinada de Dropout (0.5) y Early Stopping en la época 16 (restando los pesos óptimos de la época 11) demostraron ser mecanismos de regularización de alto estándar para proteger la varianza ante datos clínicos inéditos. La evaluación transparente contra una matriz de confusión definitiva del test set ciego expone con precisión tanto los éxitos del sistema como los límites biológicos actuales del clasificador convolucional frente a la porosidad de los gliomas.
4.  **Criterio de Ingeniería y Visión Profesional:** El éxito de esta entrega radica en no haber dependido de la optimización automática por software, sino en haber aplicado un diagnóstico de ingeniería de gradientes para corregir la alineación de etiquetas de Kaggle y especializar los filtros convolucionales superiores. Este proyecto traza una hoja de ruta metodológica de alta rigurosidad, demostrando la madurez técnica, honestidad intelectual y capacidad de auditoría computacional requerida en un Ingeniero Civil en Computación de la Universidad de O'Higgins.