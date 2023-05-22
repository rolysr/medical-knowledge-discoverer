import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
from utils.anntools import Collection
import csv


def generate_train_csv(train_path: str, output_path: str):
    
    ref = Path('./datasets/train')
    
    collection = Collection().load_dir(ref)
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
    with open('re_train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["input", "output"])
        for relation in relations.keys():
            input = str(relation)
            output = str(relations[relation])
            writer.writerow([input, output])





def get_t5_data():
    path = "re_train.csv"
    df = pd.read_csv(path)


    # simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
    df = df.rename(columns={"output":"target_text", "input":"source_text"})
    df = df[['source_text', 'target_text']]

    # T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
    df['source_text'] = df['source_text']


    train_df, test_df = train_test_split(df, test_size=0.1)
    return train_df, test_df
