# Modelo T5 aplicado a la tarea de extacción de relaciones (NER)
## Introducción

El desafío de Extracción de Conocimiento en Documentos Médicos (eHealth-KD) busca abordar la tarea de identificar y entender las entidades y relaciones presentes en textos médicos. Este informe se centra específicamente en la utilización del modelo Transformer T5 para la tarea de extracción de relaciones.

## Método

El modelo T5 (Text-to-Text Transfer Transformer) es una variante del modelo Transformer que se entrenó para convertir texto en texto. Esta configuración generalizada permite que el modelo se aplique a una variedad de tareas, simplemente cambiando el formato del texto de entrada y salida.

Para abordar la tarea de Extracción de Relaciones, decidimos entrenar el modelo T5 de una forma específica. Dada una oración, generamos un conjunto de oraciones de entrenamiento. Para cada par de entidades posible en la oración, proporcionamos al modelo T5 la oración con el par marcado, y la salida es el tipo de relación. De esta manera, el modelo es entrenado para predecir el tipo de relación entre dos entidades dadas, basándose en el contexto de la oración.

## Resultados
Precision: 0.6327014218009479
Recall: 0.6327014218009479
F1 score: 0.6327014218009479