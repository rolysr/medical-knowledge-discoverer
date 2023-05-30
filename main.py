from pathlib import Path
from models.lstm_model.classifier import Classifier
from utils.anntools import Collection
from models.lstm_model.ner_clsf import NERClassifier
import utils.score as score
from pathlib import Path
# from ontology.ontology import Ontology
# from ontology.ontology_utils import *


ref = Path('./datasets/train')
clf = Classifier()
clf.fit(ref)

for i in range(3):
    eval_ = Path('./datasets/test/')
    print(f'Evaluating testing run {i}')
    scenarios = [1, 2, 3]
    submit_ = Path(f'./results/lstm_model/test')

    clf.eval(eval_, scenarios, submit_, f'run{i}')

score.main(Path('./datasets/test'), Path('./results/lstm_model/test'), None, [1,2,3], [1,2,3], '')

# Importante: no correr este codigo sin limpiar antes la base de datos en Neo4j pues se crearian doble los nodos y las relaciones

# uri = "neo4j+s://8c1c0ab9.databases.neo4j.io"
# user = "neo4j"
# password = "F4yRbpbmrAS7ic79jwq-z-K_5gPnT7wa-LljmqnlLmA"
# app = Ontology(uri, user, password)
# app.create_database()
# # app.delGraph()
# app.close()


