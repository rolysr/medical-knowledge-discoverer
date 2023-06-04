from pathlib import Path
from models.NER.ner import NER
from utils.anntools import Collection


ner = NER()


ref = Path('./datasets/train')
collection = Collection().load_dir(ref)
ner.train(collection)

test_path = Path('./datasets/test/scenario1-main')
collection = Collection().load_dir(test_path)
results = ner.run(collection)