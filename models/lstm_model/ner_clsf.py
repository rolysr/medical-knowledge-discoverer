from utils.anntools import Collection
from pathlib import Path

from models.lstm_model.ner_utils import load_training_entities, load_testing_entities, postprocessing_labels1, get_char2idx, train_by_shape, predict_by_shape, convert_to_str_label
from models.lstm_model.base_clsf import BaseClassifier
import utils.score

from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, concatenate, Masking
from keras.losses import categorical_crossentropy
from models.lstm_model.lstm_utils import weighted_loss, detect_language, nlp_es, nlp_en
# from keras_crf import CRF
# from keras_crf import CRF
import numpy as np
import time
import json, pickle

from rich.progress import track


class NERClassifier(BaseClassifier):
    """Classifier for the name entity resolution task"""

    def __init__(self):
        BaseClassifier.__init__(self)
        self.n_tags = 6
        self.n_entities = 4
        self.encoder_tags = LabelEncoder()
        self.encoder_entities = LabelEncoder()
        self.history = None

    def train(self, collection: Collection):
        """
        Wrapper function where of the process of training is done
        """
        features, X_char, tags, entities = self.get_sentences(collection)
        X, (y_tags, y_entities) = self.preprocessing(features, (tags, entities))
        self.get_model()
        return self.fit_model((X, X_char), (y_tags, y_entities))

    def get_model(self):
        """
        Construct the neural network architecture using the keras functional api.
        `mode` is the mode where the lstm are joined in the bidirectional layer, (its not currently being used)
        """
        # input for words
        inputs = Input(shape=(None, self.n_features))
        char_in = Input(shape=(None, 10))
        # inputs of the embeddings
        emb_char = TimeDistributed(Embedding(input_dim=254, output_dim=10,
                                             input_length=10, mask_zero=True))(char_in)
        # character LSTM to get word encoding by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate((inputs, char_enc))
        x = Bidirectional(LSTM(units=32, return_sequences=True,
                               recurrent_dropout=0.1))(x)  # variational biLSTM
        x = Bidirectional(LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
       
        out1 = TimeDistributed(Dense(self.n_tags, activation="softmax"))(x)  # a dense layer as suggested by neuralNer
        out2 = TimeDistributed(Dense(self.n_entities, activation="softmax"))(x)  # a dense layer as suggested by neuralNer

        model = Model(inputs=(inputs, char_in), outputs=(out1, out2))
        model.compile(optimizer="adam", metrics=self.metrics, loss=categorical_crossentropy)
        model.summary()
        self.model = model

    def preprocessing(self, features, labels):
        """
        Handles the preprocessing step. The features and labels are converted in vectors
            and their shape is adjusted.
        """
        tags, entities = labels
        X = self.preprocess_features(features)
        y_tags = self.preprocess_labels(tags, self.encoder_tags)
        self.n_tags = y_tags[0].shape[-1]
        y_entities = self.preprocess_labels(entities, self.encoder_entities)
        self.n_entities = y_entities[0].shape[-1]
        return X, (y_tags, y_entities)

    def get_sentences(self, collection: Collection):
        """
        Giving a collection, the features and labels of its sentences are returned
        """
        features = []
        tags = []
        entities = []
        X_char = []
        self.char2idx = get_char2idx(collection)
        # embedding_vec = []
        for sentence in collection:
            feat, chars, tag, entity = load_training_entities(sentence, self.char2idx)
            features.append(feat)
            tags.append(tag)
            entities.append(entity)
            X_char.append(np.array(chars))
            # embedding_vec.append(embedding)
        return features, X_char, tags, entities

    def get_features(self, collection: Collection):
        """Giving a collection, the features of its sentences are returned"""
        features = []
        X_char = []
        # embedding_vec = []
        for sentence in collection:
            feat, chars = load_testing_entities(sentence, self.char2idx)
            features.append(feat)
            X_char.append(chars)
            # embedding_vec.append(embedding)
        return features, X_char

    def fit_model(self, X, y, plot=False):
        """
        The model is fitted. The training begins
        """
        X, X_char = X
        y_tags, y_entities = y
        
        x_shapes, x_char_shapes, yt_shapes, ye_shapes = train_by_shape(X, y_tags, y_entities,
                                                                                             X_char)
        for shape in track(x_shapes, description='Training NER...'):
            self.history = self.model.fit(
                (np.asarray(x_shapes[shape]), np.asarray(x_char_shapes[shape])),
                (np.asarray(yt_shapes[shape]), np.asarray(ye_shapes[shape])),
                epochs=10,
                verbose=0)

    def test_model(self, collection: Collection) -> Collection:
        collection = collection.clone()
        features, X_char = self.get_features(collection)
        X = self.preprocess_features(features, train=False)
        x_shapes, x_char_shapes, indices = predict_by_shape(X, X_char)
        pred_tags = []
        pred_entities = []
        for x_items, x_chars in zip(x_shapes, x_char_shapes):
            pt, pe = self.model.predict((np.asarray(x_items), np.asarray(x_chars)))
            pred_tags.extend(pt)
            pred_entities.extend(pe)
        labels_tags = self.convert_to_label(pred_tags, self.encoder_tags)
        labels_entities = self.convert_to_label(pred_entities, self.encoder_entities)
        labels = convert_to_str_label(labels_tags, labels_entities)
        entities = self.encoder_entities.classes_.tolist()
        entities.remove('None')
        postprocessing_labels1(labels, indices, collection, entities)
        return collection

    def eval(self, path: Path, submit: Path):
        folder = 'scenario2-taskA'
        scenario = path / 'scenario2-taskA'
        print(f"Evaluating on {scenario}")

        input_data = Collection().load(scenario / "input.txt")
        print(f'Loaded {len(input_data)} input sentences')
        output_data = self.test_model(input_data)

        print(f"Writing output to {submit / folder}")
        output_data.dump(submit / folder / "output.txt", skip_empty_sentences=False)

    def save_model(self, name):
        BaseClassifier.save_model(self, name)
        json.dump(self.char2idx, open(fr'resources/{name}_charmap.json', 'w'))
        pickle.dump(self.encoder_tags, open(fr'resources/{name}_tag_encoder.pkl', 'wb'))
        pickle.dump(self.encoder_entities, open(fr'resources/{name}_entity_encoder.pkl', 'wb'))

    def load_model(self, name):
        BaseClassifier.load_model(self, name)
        self.char2idx = json.load(open(fr'resources/{name}_charmap.json', 'r'))
        self.encoder_tags = pickle.load(open(fr'resources/{name}_tag_encoder.pkl', 'rb'))
        self.encoder_entities = pickle.load(open(fr'resources/{name}_entity_encoder.pkl', 'rb'))