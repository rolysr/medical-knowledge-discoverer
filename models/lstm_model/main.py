from pathlib import Path
from models.lstm_model.classifier import Classifier
from models.lstm_model.ner_clsf import NERClassifier
import utils.score as score
from pathlib import Path




def lstm():
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