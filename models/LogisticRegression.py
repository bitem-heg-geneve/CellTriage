import string

import numpy as np
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import random
import time

import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from api.sibils import MedlineFetch

nlp = spacy.load("en_core_sci_md")

# Tokenizer
parser = English()
stop_words = spacy.lang.en.stop_words.STOP_WORDS
punctuations = string.punctuation


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in mytokens
    ]

    # Removing stop words
    mytokens = [
        word for word in mytokens if word not in stop_words and word not in punctuations
    ]

    return mytokens


# Clean text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    # return text.strip().lower()
    return text.lower()


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


class BagOfWords(object):
    def __init__(self):
        self.X = []
        self.y = []

        self.vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
        self.model = LogisticRegression()
        self.pipe = Pipeline(
            [
                ("cleaner", predictors()),
                ("vectorizer", self.tfidf_vector),
                ("classifier", self.classifier),
            ]
        )
        self.pipe = None
        self.accuracy = 0
        self.precision = 0
        self.f1 = 0
        self.recall = 0
        self.train_time = 0
        self.n_splits = min([5, len(self.X)])
        self.random_state = 30481

    def train(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train).astype(int)
        start = time.time()
        accuracy_scores, f1_scores, precision_scores, recall_scores = [], [], [], []
        cv = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for train_index, test_index in cv.split(X):
            X_train, X_test, y_train, y_test = (
                X[train_index],
                X[test_index],
                y[train_index],
                y[test_index],
            )
            pipe = Pipeline(
                [
                    ("cleaner", predictors()),
                    ("vectorizer", self.tfidf_vector),
                    ("classifier", self.classifier),
                ]
            )
            pipe.fit(X_train, y_train)
            predicted = pipe.predict(X_test)
            accuracy_scores.append(metrics.accuracy_score(y_test, predicted))
            f1_scores.append(metrics.accuracy_score(y_test, predicted))
            precision_scores.append(metrics.precision_score(y_test, predicted))
            recall_scores.append(metrics.recall_score(y_test, predicted))

        end = time.time()
        self.pipe = pipe
        self.precision = round(sum(precision_scores) / len(precision_scores), 3)
        self.f1 = round(sum(f1_scores) / len(f1_scores), 3)
        self.recall = round(sum(recall_scores) / len(recall_scores), 3)
        self.accuracy = round(sum(accuracy_scores) / len(accuracy_scores), 3)
        self.train_time = round(end - start, 3)

    def predict_proba(self, texts, normalize=False):
        results = []
        if texts:
            preds = self.pipe.predict_proba(texts)[:, 1]

            if normalize:
                results = (preds - preds.min()) / (
                    preds.max() - preds.min() + 0.0000001
                )  # min-max
            else:
                results = preds
        return results

    def predict(self, texts, normalize=False):
        results = []
        if texts:
            preds = self.pipe.predict_proba(texts)[:, 1]

            if normalize:
                results = (preds - preds.min()) / (
                    preds.max() - preds.min() + 0.0000001
                )  # min-max
            else:
                results = preds
                
    def evaluate (self, X, y):
        pass
    
    def save(path):
        pass
     
    def load:
        pass

                
