import os
import csv
import pandas as pd
from re import sub
from copy import copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5

from utils.anntools import Collection



class T5:
    def __init__(self) -> None:        
        self.model = None


    def get_trained_models(self, path: str):
        '''
        Get the list of current trained models.
        '''
        return os.listdir(path)


    # OK
    def get_marked_sentence_t5_input_format(self, sentence, ent1, label1, ent2, label2):
        new_sentence = copy(sentence)
        new_sentence = new_sentence.replace(ent1, "<{0}1>{1}</{0}1>".format(label1, ent1))
        new_sentence = new_sentence.replace(ent2, "<{0}2>{1}</{0}2>".format(label2, ent2))
        return new_sentence


    # OK
    def get_t5_output_format(self, ent1, label1, ent2, label2, relation_label):
        return relation_label + "({}, {})".format(label1+"1", label2+"2")


    # OK
    def generate_t5_input_output_format(self, collection):
        dataset = []
        for sentence in collection.sentences:
            for relation in sentence.relations:
                origin = relation.from_phrase
                origin_text = origin.text.lower()
                destination = relation.to_phrase
                destination_text = destination.text.lower()

                #making the pair
                input_text = self.get_marked_sentence_t5_input_format(sentence.text, origin_text, origin.label, destination_text, destination.label)
                output_text = self.get_t5_output_format(origin_text, origin.label, destination_text, destination.label, relation.label)
                dataset.append((input_text, output_text))
        return dataset
    

    # OK
    def generate_csv(self, dataset, output_path: str):
        '''
        Generate a .csv file from the data.
        '''
        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["input", "output"])
            for data in dataset:
                input = str(data[0])
                output = str(data[1])
                writer.writerow([input, output])
    

    ## OK
    def load_csv(self, path: Path):
        '''
        Load the train data .csv
        '''
        df = pd.read_csv(path)
        
        # simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
        df = df.rename(columns={"output":"target_text", "input":"source_text"})
        df = df[['source_text', 'target_text']]

        # T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
        df['source_text'] = df['source_text']

        return df


    # OK
    def train(self, train_dataset: str, csv_file: str, outputdir: str, MAX_INPUT_TOKENS: int, MAX_OUTPUT_TOKENS: int, batch_size=8, epochs=4, use_gpu=False):
        '''
        Train the model.
        '''
        self.generate_csv(train_dataset, str(csv_file))
        df = self.load_csv(str(csv_file))
        train_df, test_df = train_test_split(df, test_size=0.1)

        model = SimpleT5()
        model.from_pretrained(model_type="t5", model_name="t5-base")
        
        print('Training...')
        model.train(train_df=train_df,
                    eval_df=test_df, 
                    source_max_token_len=MAX_INPUT_TOKENS, 
                    target_max_token_len=MAX_OUTPUT_TOKENS, 
                    batch_size=batch_size,
                    max_epochs=epochs,
                    use_gpu=use_gpu,
                    outputdir=outputdir
                )
        print('End Training')
        self.model = model
        return model
    
