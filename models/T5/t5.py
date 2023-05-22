import os
import csv
import pandas as pd
from pathlib import Path
from simplet5 import SimpleT5
from rich.progress import track
from sklearn.model_selection import train_test_split

from utils.score import main_scenario, report_main
import utils.score as score
from utils.anntools import *




class T5:
    def __init__(self) -> None:        
        self.scenarios = {
            1: ("scenario1-main", True, True),
            2: ("scenario2-taskA", True, False),
            3: ("scenario3-taskB", False, True),
        }

        self.model = None


    def get_trained_models(self, path: str):
        '''
        Get the list of current trained models.
        '''
        return os.listdir(path)
    

    def generate_train_csv(self, train_path: str, save_file: str):
        '''
        Generate a .csv file from the train data.
        '''
        collection = Collection().load_dir(Path(train_path))
        print(f"Loaded {len(collection)} sentences for fitting.")

        relations = {}

        for sentence in collection.sentences:
            for relation in sentence.relations:
                origin = relation.from_phrase
                origin_text = origin.text.lower()
                destination = relation.to_phrase
                destination_text = destination.text.lower()

                relations[
                    origin_text, origin.label, destination_text, destination.label
                ] = relation.label

        # store relations for training in re_train.csv
        with open(save_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["input", "output"])
            for relation in relations.keys():
                input = str(relation)
                output = str(relations[relation])
                writer.writerow([input, output])


    def get_csv_data(self, train_path: str):
        '''
        Get the .csv generated.
        '''
        df = pd.read_csv(train_path)

        # simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
        df = df.rename(columns={"output":"target_text", "input":"source_text"})
        df = df[['source_text', 'target_text']]

        # T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
        df['source_text'] = df['source_text']


        train_df, test_df = train_test_split(df, test_size=0.1)
        return train_df, test_df


    def train(self, train_path: str, csv_file: str):
        '''
        Train the model.
        '''
        self.generate_train_csv(train_path, csv_file)
        train_df, test_df = self.get_csv_data(csv_file)

        model = SimpleT5()
        model.from_pretrained(model_type="t5", model_name="t5-base")
        
        print('Training...')
        model.train(train_df=train_df[:5000],
                    eval_df=test_df[:100], 
                    source_max_token_len=128, 
                    target_max_token_len=50, 
                    batch_size=8,
                    max_epochs=3,
                    use_gpu=False,
                    outputdir="output"
                    )
        print('End Training')
        
        self.model = model
        
        return model


    def test_model(self, input_data: Collection):
        '''
        Generate test Collection.
        '''
        test_collection = Collection()

        for k in track(range(len(input_data.sentences)), description="Testing..."):
            sentence = input_data.sentences[k]
            test_sentence = Sentence(sentence.text)
            for keyphrase in sentence.keyphrases:
                test_sentence.keyphrases.append(keyphrase)
            for i in range(len(sentence.keyphrases)):
                for j in range(len(sentence.keyphrases)):
                    if i != j:
                        keyphrase1 = test_sentence.keyphrases[i]
                        keyphrase2 = test_sentence.keyphrases[j]
                        model_input_prediction = (keyphrase1.text, keyphrase1.label, keyphrase2.text, keyphrase2.label)
                        prediction_input = str(model_input_prediction)
                        test_sentence.relations.append(Relation(sentence, keyphrase1.id, keyphrase2.id, self.model.predict(prediction_input)))
            test_collection.sentences.append(test_sentence)
        return test_collection


    def run(self, collection, taskA, taskB):
        '''
        Its supposed to run the test example.
        '''
        collection = collection.clone()

        if taskA:
            collection = self.test_model(collection)
        if taskB:
            collection = self.test_model(collection)
        return collection


    def eval(self, path: Path, scenarios: List[int], submit: Path, run):
        '''
        Function that evals according to the baseline classifier.
        '''
        for id in scenarios:
            folder, taskA, taskB = self.scenarios[id]

            scenario = path / folder
            print(f"Evaluating on {scenario}.")

            input_data = Collection().load(scenario / "input.txt")
            print(f"Loaded {len(input_data)} input sentences.")
            output_data = self.run(input_data, taskA, taskB)

            print(f"Writing output to {submit / run / folder }")
            output_data.dump(submit / run / folder  / "output.txt", skip_empty_sentences=False)


    def main_score(self, gold: Path, submit: Path, scenarios: List[int], runs: int, prefix:str, verbose:bool=None):
        runs_data = {}

        for run in range(runs):
            run_data = {}

            if not (submit / f"run{run}").exists():
                print(f"Run {run} not found!")
                continue
            
            for id in scenarios:
                folder, skipA, skipB = self.scenarios[id]

                print(f"Scoring scenario {id} on run {run}:\n")
                run_data[folder.split("-")[0]] = main_scenario(gold / folder / "input.txt", submit / 'run{}'.format(run) / folder / "output.txt", skipA, skipB, verbose)
                print()

            runs_data[f"run{run}"] = run_data

        print()
        report_main(runs_data, prefix)
        return runs_data
    
    def main(self):

        output_path = Path('./output')
        os.makedirs(output_path, exist_ok=True)
        
        train_path = Path('./datasets/train')
        csv_file = Path('models/T5/re_train.csv')

        test_path = Path('./datasets/test')
        submit_path = Path('./results/t5/test')
        
        # get trained models
        trained_models = self.get_trained_models(output_path)

        if trained_models == []:
            # Train an get the best
            print('############## TRAIN PHASE ##############')
            model = self.train(train_path, csv_file)
            best = self.get_trained_models(output_path)[-1]
            print('------best model {}'.format(best))
            print('############## LOAD MODEL PHASE ##############')
            model.load_model("t5", output_path / best, use_gpu=False)
            self.model = model
        else:
            # Get the best trained model
            print('############## LOAD MODEL PHASE ##############')
            model = SimpleT5()
            best = trained_models[-1]
            print('------best model {}'.format(best))
            model.load_model("t5", output_path / best, use_gpu=False)
            self.model = model

        print('############## TEST PHASE ##############')
        for i in range(3):
            print('Evaluation run{}'.format(i))
            self.eval(test_path, [1, 2, 3], submit_path, 'run{}'.format(i))

        print('############## EVALUATION PHASE ##############')
        self.main_score(gold=test_path, submit=submit_path, scenarios=[1, 2, 3], runs=3, prefix='')
