from sklearn.datasets import fetch_20newsgroups

try:
    import cPickle as pickle
except:
    import pickle


class TestData(object):
    def __init__(self):
        self.categories = [
            "alt.atheism",
            "talk.religion.misc",
        ]
        # self.X_train, self.y_train, self.X_test, self.y_test = [], [], [], []

    def create(self, n_train=None, n_test=None):
        self.X_train, self.y_train = fetch_20newsgroups(
            subset="train",
            categories=self.categories,
            shuffle=True,
            random_state=42,
            remove=("headers", "footers", "quotes"),
            return_X_y=True,
        )

        self.X_test, self.y_test = fetch_20newsgroups(
            subset="test",
            categories=self.categories,
            shuffle=True,
            random_state=42,
            remove=("headers", "footers", "quotes"),
            return_X_y=True,
        )

        if n_train is not None:
            n_train = min(len(self.X_train), n_train)
            self.X_train = self.X_train[:n_train]
            self.y_train = self.y_train[:n_train]

        if n_test is not None:
            n_test = min(len(self.X_test), n_test)
            self.X_test = self.X_test[:n_test]
            self.y_test = self.y_test[:n_test]

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
