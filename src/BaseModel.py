from pathlib import Path

from multiprocessing import Pool
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import KFold
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

    def eval(self, X, y):
        y_pred = self.predict(X)
        return self.metrics(y, y_pred)

    def cv(self, X, y, kfold=5, pool=1):
        X = np.array(X)
        y = np.array(y).astype(int)

        train_predict_map = []
        y_val_map = []
        kf = KFold(kfold, shuffle=True, random_state=self.random_state)
        for fold_train, fold_val in kf.split(X):
            train_predict_map.append((X[fold_train], y[fold_train], X[fold_val]))
            y_val_map.append(y[fold_val])

        with Pool(pool) as p:
            y_pred_map = p.starmap(
                self.train_predict,
                train_predict_map,
            )

        precision, recall, accuracy, f1 = [], [], [], []
        for fold in zip(y_pred_map, y_val_map):
            precision.append(precision_score(fold[0], fold[1]))
            recall.append(recall_score(fold[0], fold[1]))
            accuracy.append(accuracy_score(fold[0], fold[1]))
            f1.append(f1_score(fold[0], fold[1]))
        return self.metrics_collape(precision, recall, accuracy, f1)

    # def __train_predict(self, X_train, y_train, X_val):
    #     return LogReg().train(X_train, y_train).predict(X_val)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
