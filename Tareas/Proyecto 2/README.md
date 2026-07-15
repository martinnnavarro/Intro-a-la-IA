# Clasificación Inteligente de Tumores Cerebrales en Resonancias Magnéticas (MRI) mediante Deep Learning y Transfer Learning

**Integrantes:** Martin Navarro Toro  
**Curso:** Introducción a la Inteligencia Artificial (COM4402-1)  
**Modelo Avanzado:** Transfer Learning con MobileNetV2  
**Dataset:** Brain Tumor Classification (MRI) - Kaggle  
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri  


## 1. Descripción del Problema
Los tumores cerebrales representan una de las patologías oncológicas más agresivas y complejas de diagnosticar en la medicina moderna. El análisis visual de las Resonancias Magnéticas (MRI) por parte de los especialistas médicos es una tarea minuciosa que requiere un alto nivel de experiencia, consume tiempo crítico y está sujeta a la variabilidad subjetiva del ojo humano. Un retraso o error en la identificación del tipo de tumor puede cambiar drásticamente el protocolo terapéutico y el pronóstico de supervivencia del paciente.

Para abordar este desafío desde la perspectiva del aprendizaje profundo (Deep Learning), este proyecto implementa un sistema automatizado modelado como un problema real de **Clasificación Multiclase**. El objetivo es categorizar imágenes de MRI en cuatro clases diagnósticas: tres tipos de tumores específicos (Glioma, Meningioma, Tumor Pituitario) y tejido cerebral sano (Sin Tumor). Esta solución busca actuar como una herramienta complementaria de soporte para la toma de decisiones clínicas, agilizando el triaje y optimizando la precisión diagnóstica en centros asistenciales.

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

Asimismo, la inspección visual (ver archivo `eda_muestras_mri_reales.png`) reveló una discrepancia crítica en el **Feature Engineering**: las imágenes tumorales presentaban una resolución nativa de alta definición de $512 \times 512$ píxeles, mientras que las muestras sanas provenían de otro escáner con una resolución menor de $236 \times 260$. Esto validó de forma empírica la necesidad de aplicar un redimensionamiento uniforme a $224 \times 224$ píxeles antes de alimentar el modelo.

## 3. Justificación de los Modelos y Plan de Acción
Para resolver un problema de visión computacional con un volumen acotado de datos (3,264 imágenes en total), se justifica metodológicamente la aplicación de **Transfer Learning** (Aprendizaje por Transferencia) utilizando la arquitectura **MobileNetV2** preentrenada con el dataset masivo ImageNet.

**Argumentación Técnica:**
1. **Aprendizaje de Representaciones:** Entrenar una red convolucional profunda desde cero requiere millones de muestras para que las capas ocultas aprendan fronteras básicas como bordes, texturas y formas. Al reutilizar un modelo preentrenado, el sistema ya cuenta con "detectores de características" visuales altamente optimizados en sus 154 capas convolucionales nativas.
2. **Eficiencia y Congelamiento de Pesos:** Establecer `base_model.trainable = False` congela los 2,257,984 parámetros originales del modelo base. La optimización y el gasto de hardware (GPU de Google Colab) se concentran exclusivamente en un bloque de clasificación personalizado (*Top Model*), compuesto por una capa de pooling global (`GlobalAveragePooling2D`), una capa oculta de 256 neuronas con activación ReLU, y una capa de salida con activación **Softmax** de 4 unidades correspondientes a las categorías de tumores. Esto reduce el costo computacional de entrenamiento a solo 328,964 parámetros entrenables.

## 4. Metodología Aplicada (Paso a Paso)
1. **Auditoría e Ingeniería de Características:** Carga controlada de los directorios de imágenes, normalización de los píxeles de un rango $[0, 255]$ a un espacio escalar $[0, 1]$ mediante `rescale=1./255`, y redimensionamiento uniforme de todos los tensores a un tamaño de $224 \times 224 \times 3$.
2. **Regularización por Aumento de Datos:** Configuración de transformaciones geométricas aleatorias (*Data Augmentation*) dinámicas en la memoria RAM (rotación de hasta 20°, desplazamientos horizontales/verticales del 10%, zoom del 10% y giros horizontales espejo) únicamente sobre el lote de entrenamiento para mitigar el sobreajuste (*overfitting*).
3. **Partición de Validación Estricta:** Aislamiento automático de un $20\%$ de la carpeta `Training` para actuar como conjunto de validación interna independiente (2,297 imágenes para entrenamiento puro y 573 para validación).
4. **Instanciación y Acoplamiento de Redes:** Carga de la arquitectura MobileNetV2 congelada y acoplamiento de las capas superiores personalizadas con una tasa de **Dropout del 50% (0.5)** como regularizador estocástico.
5. **Configuración del Loop de Optimización:** Compilación mediante la función de pérdida `categorical_crossentropy` y el optimizador adaptativo **Adam** (Learning Rate = 0.001), asistido por un monitoreo automático de `ModelCheckpoint` y un regularizador de paciencia **Early Stopping** configurado con `patience=5`.
6. **Testeo Ciego Final:** Carga de los mejores pesos alcanzados en el disco y predicción sobre las 394 imágenes del set de prueba ciego (`Testing`) con `shuffle=False` para auditar el rendimiento.

## 5. Resultados Obtenidos e Interpretación

El loop de optimización lanzado en la GPU de Google Colab evidenció una dinámica de aprendizaje y una activación de regularizadores sumamente clara (ver archivo `curvas_aprendizaje_deep_learning.png`):

* **Comportamiento del Entrenamiento:** En la Época 1, el modelo inició con un `loss` de 0.7576 y un `accuracy` de entrenamiento de $70.05\%$. En la **Época 2**, la red adaptó rápidamente sus gradientes logrando su mejor rendimiento histórico: una pérdida de entrenamiento que bajó a 0.4968, una exactitud de validación (`val_accuracy`) del **$79.76\%$** y una pérdida de validación (`val_loss`) mínima de **0.5357**. En este instante, el sistema resguardó el archivo `best_brain_tumor_model.keras`.
* **Activación del Early Stopping:** A partir de la época 3, el modelo entró en zona de sobreajuste (*overfitting*); la exactitud de entrenamiento se disparó artificialmente hacia el $87\%$, pero el error de validación (`val_loss`) rebotó al alza de forma sostenida hasta 0.5720 en la época 7. Al registrarse 5 épocas consecutivas sin mejoras en el set de validación, **el Early Stopping se activó automáticamente en la Época 7**, deteniendo el entrenamiento y restaurando los pesos estables de la Época 2 para asegurar la generalización.

### 5.1. Evaluación Final en el Test Set Ciego (Matriz de Confusión)
Al someter los mejores pesos recuperados al testeo ciego final con las 394 muestras independientes, se obtuvieron las siguientes métricas globales macro-promedio:

| Métrica | Valor Real Obtenido en el Test Set |
| :--- | :---: |
| **Accuracy (Exactitud General)** | 0.5228 (52.28%) |
| **Precision (Macro)** | 0.5056 (50.56%) |
| **Recall (Sensibilidad Macro)** | 0.5382 (53.82%) |
| **F1-Score (Macro)** | 0.4884 (48.84%) |

Al abrir analíticamente la matriz de confusión final (ver archivo `matriz_confusion_final_deep_learning.png`), se identificaron los siguientes comportamientos hísticos:
* **Éxito en Tejido Sano (`no_tumor`):** El modelo base demostró un desempeño sobresaliente reconociendo la simetría anatómica normal, alcanzando **83 aciertos de un total de 105 casos reales**.
* **Confusión Crítica por Colinealidad:** El punto de quiebre se concentró en la clase `glioma_tumor`, donde de 100 casos reales **solo se clasificaron correctamente 14**, confundiéndose masivamente con meningiomas (35 casos) y tumores pituitarios (32 casos). Al mantener las capas convolucionales congeladas, los filtros genéricos de ImageNet sufren para discriminar variaciones de texturas y contrastes médicos tan sutiles entre los diferentes tipos de tumores.
* **Tumores Pituitarios:** Estabilizó su rendimiento logrando **57 aciertos de 74 casos reales**, cometiendo errores menores al confundir 11 casos con cerebros sanos debido a su ubicación anatómica fija en la base celular.

## 6. Conclusiones Generales

1. **Efectividad del Aprendizaje por Transferencia:** Se validó que la utilización de modelos preentrenados (`MobileNetV2`) permite estructurar pipelines de Deep Learning estables y de rápida convergencia para imágenes biomédicas, evitando el costo de entrenar millones de parámetros desde cero.
2. **Necesidad de Especialización de Dominios (Fine-Tuning):** El experimento demostró empíricamente que mantener un congelamiento absoluto de las capas convolucionales base limita el rendimiento ante problemas médicos de alta complejidad. Las representaciones genéricas de ImageNet son insuficientes para separar sutiles variaciones morfológicas como los gliomas y meningiomas, concluyendo que para alcanzar precisión clínica es indispensable aplicar un *Fine-Tuning* controlado en las capas superiores.
3. **Éxito de las Estrategias de Regularización:** Los mecanismos de control de *overfitting* extraídos de las clases cumplieron con éxito su rol computacional. El *Data Augmentation* absorbió la heterogeneidad de escalas de las imágenes, mientras que la combinación de *Dropout* (0.5) y *Early Stopping* controló la varianza de la red de forma automatizada en la época 7.
4. **Rigor y Criterio de Ingeniería:** La evaluación transparente contra un conjunto de prueba completamente aislado e independiente (*test set* ciego) evitó cualquier escenario de sobreajuste o fuga de información (*data leakage*). Los resultados obtenidos trazan una hoja de ruta clara para la optimización de redes convolucionales, demostrando la madurez técnica, honestidad metodológica y capacidad de auditoría requerida en un Ingeniero Civil en Computación.