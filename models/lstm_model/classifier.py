from typing import List
from pathlib import Path

from utils.anntools import Collection
from models.lstm_model.ner_clsf import NERClassifier
from models.lstm_model.re_clsf import REClassifier

import utils.score

class Classifier:
    """
    Classifier for the main task.
    It wraps the name entity classifier and the relation extractor classifier
    """

    def __init__(self):
        self.ner_classifier = NERClassifier()
        self.re_classifier = REClassifier()

    scenarios = {
        1: ("scenario1-main", True, True),
        2: ("scenario2-taskA", True, False),
        3: ("scenario3-taskB", False, True),
    }

    def fit(self, path: Path):
        """Does all the process of training in the classifiers"""
        collection = Collection().load_dir(path)

        print(f"Loaded {len(collection)} sentences for fitting.")
        print('Starting ner classifier training')
        self.ner_classifier.train(collection)
        print('Starting re classifier training')
        self.re_classifier.train(collection)
        print(f"Training completed.")

    def eval(self, path: Path, scenarios: List[int], submit: Path, run):
        """Function that evals according to the baseline classifier"""
        # Its not changed 
        for id in scenarios:
            folder, taskA, taskB = self.scenarios[id]

            scenario = path / folder
            print(f"Evaluating on {scenario}.")

            input_data = Collection().load(scenario / "input.txt")
            print(f"Loaded {len(input_data)} input sentences.")
            output_data = self.run(input_data, taskA, taskB)

            print(f"Writing output to {submit / run / folder }")
            output_data.dump(submit / run / folder  / "output.txt", skip_empty_sentences=False)


    def run(self, collection, taskA, taskB):
        """Its supposed to run the test example"""
        # gold_keyphrases, gold_relations = self.model
        collection = collection.clone()

        if taskA:
            collection = self.ner_classifier.test_model(collection)
        if taskB:
            collection = self.re_classifier.test_model(collection)
        return collection