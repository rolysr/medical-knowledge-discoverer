# Modelo BERT aplicado a la tarea de reconocimiento de entidades (NER)

## Introducción
En este proyecto de descubrimiento de conocimiento médico, se aborda la tarea de reconocimiento de entidades nombradas (NER) utilizando el modelo BERT (Bidirectional Encoder Representations from Transformers). La NER es una tarea crucial en el procesamiento del lenguaje natural para identificar y clasificar entidades relevantes en textos médicos, como enfermedades, síntomas, medicamentos, etc. Para este proyecto, se utilizó el conjunto de datos eHealthKD2021, que proporciona información médica relevante para entrenar y evaluar el modelo.

En la actualidad, BERT se considera uno de los modelos de lenguaje más avanzados en NLP debido a su capacidad para capturar el contexto bidireccional de las palabras en una oración o secuencia de texto. Esto permite comprender mejor las relaciones y significados de las palabras en un contexto específico, lo que resulta beneficioso para tareas como la NER en el campo médico.

## Método:
En este proyecto, se siguió el siguiente método para ajustar el modelo BERT y abordar la tarea de NER en el conjunto de datos médicos eHealthKD2021:

1. Preprocesamiento de datos: Los datos de eHealthKD2021 se prepararon para que fueran compatibles con el formato de entrada requerido por BERT. Se tokenizó el texto y se agregaron los marcadores especiales [CLS] (inicio) y [SEP] (final). Además, se aplicaron técnicas de tokenización para dividir las palabras en subunidades más pequeñas y mejorar la capacidad del modelo para identificar entidades.

2. Ajuste del modelo BERT: Se utilizó una implementación preentrenada de BERT y se ajustaron los hiperparámetros para adaptar el modelo a la tarea de NER con los datos médicos. Se seleccionó la arquitectura BERT apropiada y se configuraron los hiperparámetros, como la tasa de aprendizaje y el tamaño del lote. El modelo se entrenó utilizando el conjunto de datos de entrenamiento durante 2 epochs (épocas) para ajustar los pesos y mejorar su rendimiento en la tarea de NER.

3. Evaluación del modelo: Una vez ajustado el modelo, se evaluó su desempeño utilizando el conjunto de datos de prueba. Se utilizaron métricas estándar de evaluación de NER, como precisión, recobrado y F1, para medir la calidad del modelo en la identificación de entidades. 

## Resultados:
Los resultados obtenidos en este proyecto fueron los siguientes:
- Precisión: 1.0
- Recobrado: 0.8375
- F1: 0.4557823129251701
Estos resultados indican que todas las entidades identificadas por el modelo son correctas (precisión de 1.0). Sin embargo, el recobrado de 0.8375 sugiere que no se logró identificar todas las entidades presentes en los datos de prueba. Esta discrepancia puede deberse a la complejidad del lenguaje médico y a las dificultades asociadas con ciertas entidades que pueden ser más difíciles de identificar.