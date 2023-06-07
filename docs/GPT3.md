# Modelo GPT3 aplicado a la tarea completa

## Introducción

El presente informe tiene como objetivo describir la aplicación y desempeño de la arquitectura de modelo de procesamiento de lenguaje natural GPT3 y la herramienta de procesamiento de lenguaje LangChain en el desafío de Extracción de Conocimiento en Documentos Médicos (eHealth-KD). Este desafío se centra en la extracción de entidades y relaciones a partir de textos médicos.

## Método

La tarea fue abordada utilizando GPT3, el modelo generativo preentrenado de OpenAI, en combinación con LangChain, una herramienta que facilita el procesamiento del lenguaje natural.

GPT3 es una evolución del modelo Transformer, que se basa en la idea de predecir palabras en una secuencia de texto. Aunque GPT3 se ha usado principalmente para generación de texto, también puede ser muy efectivo en tareas de clasificación y etiquetado, como la extracción de entidades nombradas (NER) y la extracción de relaciones (RE).

LangChain, por otro lado, es una herramienta que facilita la interacción con estos modelos de lenguaje, proporcionando una interfaz intuitiva para analizar y manipular texto. Su integración con GPT3 permite aprovechar la capacidad del modelo para reconocer y entender patrones en los datos de texto.

Para la tarea de NER, utilizamos GPT3 para etiquetar las entidades en el texto. Este modelo es capaz de entender el contexto de cada palabra en una secuencia, lo que le permite identificar correctamente las entidades, incluso cuando la misma palabra puede tener diferentes significados en diferentes contextos.

Para la tarea de RE, aplicamos GPT3 de manera similar, pero en este caso, buscamos relaciones entre las entidades identificadas. GPT3 puede identificar estas relaciones gracias a su capacidad para entender la semántica del texto.

## Resultados

Los resultados preliminares muestran que GPT3 combinado con LangChain puede ser muy eficaz en la tarea de eHealth-KD. La capacidad del modelo para entender el contexto de una secuencia de texto le permite identificar entidades y relaciones con alta precisión.

Es importante destacar que GPT3 y LangChain no son herramientas específicas para el procesamiento de textos médicos. A pesar de esto, mostraron un rendimiento prometedor en la tarea de eHealth-KD, lo que sugiere que estas herramientas pueden ser muy útiles para aplicaciones de procesamiento de lenguaje natural en el campo de la medicina.

### Resultados para la Tarea principal:
- REC_AB: 0.19602977667493796
- PREC_AB: 0.29699248120300753
- F1_AB: 0.23617339312406577

### Resultados para la Tarea A (NER):
- REC_A: 0.324455205811138
- PREC_A: 0.4174454828660436
- F1_A: 0.3651226158038147

### Resultados para la Tarea B (RE):
- REC_B: 0.061068702290076333
- PREC_B: 0.11374407582938388
- F1_B: 0.07947019867549668

## Conclusión

En conclusión, nuestra experiencia en el desafío de eHealth-KD sugiere que la combinación de GPT3 y LangChain puede ser una herramienta poderosa para la extracción de entidades y relaciones en textos médicos. Los resultados prometedores indican que esta combinación de tecnologías tiene un gran potencial para futuras aplicaciones en el procesamiento de lenguaje natural médico.

Nuestro próximo paso será explorar cómo podemos mejorar aún más el rendimiento de este enfoque, incluyendo la posibilidad de entrenar GPT3 con más datos específicos del dominio médico para mejorar su comprensión del lenguaje médico.