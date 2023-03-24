import warnings

import numpy as np

# import scispacy
import spacy

# from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import ndjson

from BaseModel import BaseModel

warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_sci_md")  # scispacy


def spacy_tokenizer(text):
    tokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in nlp(text)
        if not word.is_stop and word.is_alpha
    ]
    return tokens


class LogReg(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer(tokenizer=spacy_tokenizer)),
                ("classifier", LogisticRegression()),
            ]
        )

    def train(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X, normalize=False):
        if len(X) > 0:
            y_pred = self.pipeline.predict_proba(X)

            if normalize:
                y_pred = (y_pred - y_pred.min()) / (
                    y_pred.max() - y_pred.min() + 0.0000001
                )  # min-max
        return y_pred

    def predict(self, X, normalize=False):
        if len(X) > 0:
            y_pred = self.pipeline.predict(X)
        return y_pred

    def train_predict(self, X_train, y_train, X_val):
        return LogReg().train(X_train, y_train).predict(X_val)


def test():
    from pprint import pprint
    from time import time

    from TestData import TestData

    DATA_FP = "data/test/TestData_100_50.pkl"
    MODEL_FP = "models/test/LogReg_100_50.pkl"

    # data = TestData()
    # data.create(n_train=100, n_test=50)
    # data.save(DATA_FP)
    print("\nLoad data")
    data = TestData.load(DATA_FP)
    print(f"n_train: {len(data.y_train)}, n_test: {len(data.y_test)}")

    print("\nCrossvalidate model")
    t0 = time()
    pprint(LogReg().cv(data.X_train, data.y_train, kfold=5, pool=3))
    print(f"Done in {time() - t0:.3f}s")

    print("\nTrain model and save to disk")
    t0 = time()
    model = LogReg().train(data.X_train, data.y_train)
    model.save(MODEL_FP)
    print(f"Done in {time() - t0:.3f}s")

    print("\nLoad model from disk and evaluate on test data")
    t0 = time()
    m2 = LogReg.load(MODEL_FP)
    pprint(m2.eval(data.X_test, data.y_test))
    print(f"Done in {time() - t0:.3f}s")


if __name__ == "__main__":
    test()
