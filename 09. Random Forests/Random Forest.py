import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def _gini(self, y):
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _best_split(self, X, y):
        n, p = X.shape
        features = np.random.choice(
            p,
            self.max_features if self.max_features else p,
            replace=False
        )

        best_gini = float("inf")
        best = None

        for f in features:
            for t in np.unique(X[:, f]):
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                g = (len(left) * self._gini(left) + len(right) * self._gini(right)) / n
                if g < best_gini:
                    best_gini = g
                    best = (f, t)

        return best

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples:
            self.label = Counter(y).most_common(1)[0][0]
            return

        split = self._best_split(X, y)
        if split is None:
            self.label = Counter(y).most_common(1)[0][0]
            return

        self.feature, self.threshold = split
        idx = X[:, self.feature] <= self.threshold
        self.left = DecisionTree(self.max_depth, self.min_samples, self.max_features)
        self.right = DecisionTree(self.max_depth, self.min_samples, self.max_features)
        self.left.fit(X[idx], y[idx], depth + 1)
        self.right.fit(X[~idx], y[~idx], depth + 1)

    def predict_one(self, x):
        if self.label is not None:
            return self.label
        if x[self.feature] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)


class RandomForest:
    def __init__(self, n_trees=50, max_depth=5, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n = len(y)
        self.oob_votes = [Counter() for _ in range(n)]

        for _ in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)
            oob = set(range(n)) - set(idx)

            tree = DecisionTree(self.max_depth, max_features=self.max_features)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

            for i in oob:
                self.oob_votes[i][tree.predict_one(X[i])] += 1

    def predict(self, X):
        preds = []
        for x in X:
            votes = [t.predict_one(x) for t in self.trees]
            preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(preds)

    def oob_score(self, y):
        preds = []
        valid = []
        for i, vote in enumerate(self.oob_votes):
            if len(vote) > 0:
                preds.append(vote.most_common(1)[0][0])
                valid.append(y[i])
        return np.mean(np.array(preds) == np.array(valid))


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForest(n_trees=50, max_depth=6, max_features=4)
    rf.fit(X_train, y_train)

    print("Test accuracy:", np.mean(rf.predict(X_test) == y_test))
    print("OOB accuracy :", rf.oob_score(y_train))
