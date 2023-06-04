from typing import List
from pathlib import Path

from utils.anntools import Collection
from models.lstm_model.ner_clsf import NERClassifier



class NER:
    def __init__(self):
        self.ner_classifier = NERClassifier()


    def train(self, collection: Collection):
        """Does all the process of training."""
        self.ner_classifier.train(collection)


    # def eval(self, path: Path, scenarios: List[int], submit: Path, run):

    #     input_data = Collection().load(..., "input.txt")
    #     print(f"Loaded {len(input_data)} input sentences.")
    #     output_data = self.run(input_data)


    def run(self, collection: Collection):
        collection = collection.clone()

        collection = self.ner_classifier.test_model(collection)
        return collection