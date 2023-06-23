# Propuesta de solución para NER con T5 y Spacy

El archivo NER_T5_spacy.ipynb presenta la implementación del pipeline propuesto para abordar la subtarea A, que se centra en el reconocimiento y clasificación de entidades.

Para resolver este problema, se divide en dos partes fundamentales:

1- Detección de entidades en una oración: En este paso, se utiliza la potente biblioteca de Python llamada Spacy. Se descargan los módulos necesarios para trabajar con los idiomas inglés y español. Spacy proporciona una lista de términos en la oración junto con sus respectivas clasificaciones. Sin embargo, las clasificaciones predeterminadas de Spacy no coinciden exactamente con las categorías de nuestro modelo (Concepto, Acción, Referencia, Predicado). Por lo tanto, se modificaron las etiquetas de entidad predefinidas en Spacy para asegurar la coherencia con nuestras necesidades.

2- Clasificación de tipo de entidad: En este segundo paso, se emplea el modelo Transformer (text-to-text) llamado T5, que se entrena con cinco epoch para lograr la capacidad de clasificar las entidades según su tipo.

El pipeline incluye la evaluación de la efectividad del modelo de manera separada y conjunta. Es decir, mediante el cálculo de la precisión, el recobrado y la puntuación F1, se pueden obtener estadísticas sobre la precisión de Spacy en el paso 1 y la precisión de T5 en el paso 2. Luego, se combinan ambos métodos para resolver la subtarea A en su totalidad y se obtiene la precisión total del modelo.

## Bibliografía

-Vicomtech at eHealth-KD Challenge 2021: Deep Learning Approaches to Model Health-related Text in Spanish

- UH-MMM at eHealth-KD Challenge 2021.

- IXA at eHealth-KD Challenge 2021: Generic Sequence Labelling as Relation Extraction Approach

- Documentación oficial de [Spacy](https://spacy.io/api/doc).

- Documentación oficial de [T5](https://huggingface.co/transformers/v4.11.3/model_doc/t5.html).
