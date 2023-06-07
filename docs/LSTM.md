# Modelo LSTM aplicado a la tarea principal, NER y RE
Nuestra solución para ambas tareas se basa en Redes Neuronales Recurrentes (RNN) o, más precisamente, en Memoria a Largo Plazo Bidireccional (BiLSTM) como codificadores contextuales y capas densas como la arquitectura del decodificador de etiquetas del modelo. Esta arquitectura es elegida debido a la estructura secuencial de la entrada y es ampliamente utilizada en la literatura para abordar el problema del Reconocimiento de Entidades Nombradas (NER). El sistema utiliza información de etiquetas POS (Part-of-Speech tag), relaciones de dependencia, representaciones a nivel de caracteres, así como incrustaciones contextuales. La tarea de Extracción de Relaciones (RE) se aborda de manera pareada, codificando la información sobre la oración y el par de entidades dado utilizando estructuras sintácticas derivadas del árbol de análisis de dependencia. Además, se utilizó un tipo especial de relación para codificar la relación entre pares de entidades no relacionadas.

# Descripción del sistema
La solución propuesta resuelve ambas tareas de manera separada y secuencial. Por lo tanto, se entrenaron modelos independientes con diferentes arquitecturas y características para resolver los problemas de NER y RE. La principal distinción entre las dos arquitecturas surge del tipo de problema que resuelven. La primera tarea se plantea como un problema de predicción de etiquetas que toma el texto sin procesar de una oración como entrada y genera dos secuencias de etiquetas independientes: una en el sistema de etiquetas BILOUV para la predicción de entidades y otra con las etiquetas correspondientes a cada tipo de entidad (Concepto, Acción, Referencia, Predicado). 
La clasificación del esquema de etiquetas BILOUV corresponde a Begin, para el inicio de una entidad; Inner, para el token en medio; Last para el token final; Unit, para representar entidades de un solo token; Other para representar tokens que no pertenecen a ninguna entidad, y la etiqueta oVerlapping se utiliza para lidiar con tokens que pertenecen a múltiples entidades. Por otro lado, la segunda tarea se aborda como una serie de consultas por pares entre las entidades presentes en la oración objetivo, orientada a identificar las relaciones relevantes entre las entidades previamente extraídas.
Teniendo en cuenta las características multilingües de la tarea, el proceso de extracción de características de las características sintácticas se maneja en dos fases. En la primera, la oración de entrada se clasifica por su idioma utilizando un modelo preentrenado de FastText para la identificación del idioma [3][2]. Posteriormente, en la segunda fase, se utilizaron dos modelos diferentes de Spacy (https://spacy.io/) dependiendo del idioma de la oración (es core news sm para español y en core web sm para inglés). Estos modelos se utilizaron para extraer características como la etiqueta POS, el árbol de análisis de dependencia y la etiqueta de dependencia.

## Resultados:
scenario1_f1:0.33343789209535757
scenario1_precision:0.36909722222222224
scenario1_recall:0.3040617848970252
scenario1_best:run2
scenario2_f1:0.5598455598455597
scenario2_precision:0.5583058305830583
scenario2_recall:0.5613938053097345
scenario2_best:run2
scenario3_f1:0.04381846635367762
scenario3_precision:0.06451612903225806
scenario3_recall:0.03317535545023697
scenario3_best:run2