El archivo NER_T5_spacy.ipynb contiene el pipeline de la propuesta de solucion a la subtarea A, es decir, reconocimiento de entidades y su clasificacion.
Para llegar a esta solucion se divide el problema en 2 partes:
1- Dada una oracion(en ingles o espanol) detectar las entidades.
2- Dada una entidad, decir el tipo de esta.

Para el primer paso nos apoyamos en la libreria Spacy de Python y descargamos los modulos de ambos idiomas. Esta libreria ya te devuelve una lista de todos los terminos de la oracion con sus respectivas clasificaciones.
Estas clasificaciones no son las mismas que las de nuestro modelo(Concept, Action, Reference, Predicate) asi que modificamos las posibles entidades de Spacy que nos pudieran hacer falta.

Luego, para el segundo paso utilizamos el modelo simple T5 y lo entrenamos(con 5 epoch) para que pudiera se capaz de clasificar las entidades.

En el pipeline se calcula la efectividad de nuestro modelo de forma separada y junta, es decir, calculando la precision, el recobrado y la puntuación F1 podemos ver las estadisticas de cuan exacto es spacy para el paso 1 y cuanto lo es T5 para el paso 2. Luego, se unen ambos metodos para resolver la subtarea A como un todo.