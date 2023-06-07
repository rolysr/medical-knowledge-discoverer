import pandas as pd


from pathlib import Path
import sys
sys.path.append('C:/Users/Marie/Documents/4to/S1/ML/proyecto/medical-knowledge-discoverer/utils')
from anntools import Collection
import csv

from transformers import BertTokenizerFast, BertForTokenClassification


collection = Collection()
for fname in Path("C:/Users/Marie/Documents/4to/S1/ML/proyecto/medical-knowledge-discoverer/datasets/train").rglob("*.txt"):
    collection.load(fname)


tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(label_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(label_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def a(collection):
    lb = []
    txt = []
    for sentence in collection.sentences:
        new_sentence = sentence.text.replace('(','').replace(')','')
        txt.append(new_sentence)
        labels = [-100 for i in range(len(new_sentence.split(' ')))]
        keyphrases = sentence.keyphrases
        sentence_list = new_sentence.split(' ')
        count = 0
        k = 0
        for keyphrase in sentence.keyphrases:
            index = new_sentence.find(keyphrase.text)   
            while count + k < index:
                count += len(sentence_list[k])
                k+=1
            keyphrases_list = keyphrase.text.split(' ')
            if k < len(sentence_list) and sentence_list[k] == keyphrases_list[0]:
                labels[k]= 'B-' + keyphrase.label 
                count += len(keyphrases_list[0])
                i = 1
                while i < len(keyphrases_list):
                    labels[k + i] = 'I-' + keyphrase.label 
                    count += len(keyphrases_list[i])
                    i += 1
                k +=i
        lb.append(labels.copy())

    print((lb[0]))

    texts = [tokenizer(i,
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]

    print(texts[0])
    labels = [align_label(i,j) for i,j in zip(txt, lb)]
    print(labels[0])
    
                

a(collection)