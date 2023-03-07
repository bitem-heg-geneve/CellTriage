from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pickle


class BaseModel(object):
    random_state = 1234

    def __init__(self) -> None:
        pass

    def metrics(self, y, y_pred):
        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }

    def metrics_collape(self, precision=[], recall=[], accuracy=[], f1=[]):
        return {
            "precision": {"avg": np.mean(precision), "std": np.std(precision)},
            "recall": {"avg": np.mean(recall), "std": np.std(recall)},
            "accuracy": {"avg": np.mean(accuracy), "std": np.std(accuracy)},
            "f1": {"avg": np.mean(f1), "std": np.std(f1)},
        }

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
