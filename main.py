import os
import typer
from pathlib import Path
from simplet5 import SimpleT5
from rich.progress import track
from sklearn.model_selection import train_test_split

from models.T5.t5 import T5
from models.NER.ner import NER
import utils.score as score
from utils.anntools import Collection
from models.lstm_model.classifier import Classifier
from ontology.ontology import Ontology
from ontology.ontology_utils import *





app = typer.Typer()


@app.command()
def lstm(

):
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



@app.command()
def neo4j(
    

):
    # Importante: no correr este codigo sin limpiar antes la base de datos en Neo4j pues se crearian doble los nodos y las relaciones

    # uri = "neo4j+s://8c1c0ab9.databases.neo4j.io"
    # user = "neo4j"
    # password = "F4yRbpbmrAS7ic79jwq-z-K_5gPnT7wa-LljmqnlLmA"
    # app = Ontology(uri, user, password)
    # app.create_database()
    # # app.delGraph()
    # app.close()
    ...

    # Mostrar ejemplo: causas de covid-19
    uri = "neo4j+s://8c1c0ab9.databases.neo4j.io"
    user = "neo4j"
    password = "F4yRbpbmrAS7ic79jwq-z-K_5gPnT7wa-LljmqnlLmA"
    app = Ontology(uri, user, password)
    # app.create_database()
    enfermedad = 'covid-19'
    app.findQuery(enfermedad)
    app.close()



@app.command()
def t5_model(
    output_path = typer.Option('./output', '-o', '--output'),
    train_path = typer.Option('./datasets/train', '-t', '--train-path'),
):
    # OUTPUT
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    # TRAIN PATH
    train_path = Path(train_path)
    csv_train_file = './re_train.csv'

    # T5 MODEL
    t5 = T5()

    # GENERATE TRAIN DATA
    test_collection = Collection().load_dir(train_path)
    train_dataset = t5.generate_t5_input_output_format(test_collection)
    MAX_INPUT_TOKENS = max([len(data[0]) for data in train_dataset])
    MAX_OUTPUT_TOKENS = max([len(data[1]) for data in train_dataset])
    t5.generate_csv(train_dataset, str(csv_train_file))
    df = t5.load_csv(str(csv_train_file))
    train_df, test_df = train_test_split(df, test_size=0.1)

    # TRAIN
    t5.train(train_df, test_df, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, output_path)

    # GET TRAINED MODEL
    trained_model = os.path.join(str(output_path), os.listdir(str(output_path))[0])
    t5.load(trained_model)



@app.command()
def ner_model(
    train_path = typer.Option('./datasets/train', '-t', '--train-path'),
):
    # NER MODEL
    ner = NER()
    train_path = Path(train_path)

    # TRAINING NER MODEL
    train_collection = Collection().load_dir(train_path)
    ner.train(train_collection)



# EVALUATION
def eval(test_collection: Collection, ner_collection: Collection, model: SimpleT5, t5: T5):
    
    CORRECT, MISSING, SPURIOUS, INCORRECT = 0, 0, 0, 0

    for sentences in track(zip(test_collection.sentences, ner_collection.sentences), description='evaluating...'):
        test_sentence, ner_sentence = sentences
        
        test = {}
        for test_relation in test_sentence.relations:
            origin = test_relation.from_phrase
            origin_text = origin.text.lower()
            destination = test_relation.to_phrase
            destination_text = destination.text.lower()

            input_text = t5.get_marked_sentence_t5_input_format(test_sentence.text, origin_text, origin.label, destination_text, destination.label)
            output_text = t5.get_t5_output_format(origin_text, origin.label, destination_text, destination.label, ner_relation.label)
            
            test[test_relation] = output_text

        results= {}
        for ner_relation in ner_sentence.relations:
            origin = ner_relation.from_phrase
            origin_text = origin.text.lower()
            destination = ner_relation.to_phrase
            destination_text = destination.text.lower()

            #making the pair
            input_text = t5.get_marked_sentence_t5_input_format(ner_sentence.text, origin_text, origin.label, destination_text, destination.label)

            results[ner_relation] = model.predict(input_text)[0]
        
        
        for i in test.copy():
            if results.get(i) is not None:
                if results[i] == test[i]:
                    CORRECT += 1
                    results.pop(i)
                    test.pop(i)
                else:
                    INCORRECT += 1
                    results.pop(i)
                    test.pop(i)
        
        SPURIOUS += len(results)
        MISSING += len(test)


    return CORRECT, MISSING, SPURIOUS, INCORRECT


@app.command()
def main_eval(
    trained_model_path = typer.Option(..., '-tr', '--trained'),
    train_path = typer.Option('./datasets/train', '-tp', '--train-path'),
    test_path = typer.Option('./datasets/test/scenario1-main', '-ts', '--test'),

):
    # T5 MODEL
    t5 = T5()
    model = t5.load(trained_model_path)

    # NER MODEL
    ner = NER()

    # TRAINING NER MODEL
    train_path = Path(train_path)
    train_collection = Collection().load_dir(train_path)
    ner.train(train_collection)

    # TEST
    test_path = Path(test_path)
    test_collection = Collection().load_dir(test_path)

    # EVALUATE NER
    ner_collection = ner.run(test_collection)

    CORRECT, MISSING, SPURIOUS, INCORRECT = eval(test_collection, ner_collection, model, t5)

    # SHOW RESULTS
    precision = CORRECT / (CORRECT + MISSING + INCORRECT)
    recall = CORRECT / (CORRECT + SPURIOUS + INCORRECT)
    f1 = (2 * precision * recall) / (precision + recall)

    print("Precision:", precision)
    print('Recall:', recall)
    print('F1 score:', f1)


if __name__ == '__main__':
    app()