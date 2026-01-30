import numpy as np


class DecisionStump:
    def fit(self, X, y, w):
        n, d = X.shape
        self.feature = 0
        self.threshold = 0.0
        self.polarity = 1
        best_error = float("inf")

        for j in range(d):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                for p in [1, -1]:
                    preds = p * np.sign(X[:, j] - t)
                    preds[preds == 0] = 1
                    err = np.sum(w[preds != y])
                    if err < best_error:
                        best_error = err
                        self.feature = j
                        self.threshold = t
                        self.polarity = p

    def predict(self, X):
        preds = self.polarity * np.sign(X[:, self.feature] - self.threshold)
        preds[preds == 0] = 1
        return preds


class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        w = np.ones(n) / n

        for _ in range(self.T):
            stump = DecisionStump()
            stump.fit(X, y, w)
            preds = stump.predict(X)

            err = np.sum(w[preds != y])
            err = max(err, 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            w *= np.exp(-alpha * y * preds)
            w /= np.sum(w)

            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        agg = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            agg += alpha * model.predict(X)
        return np.sign(agg)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = AdaBoost(T=20)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print("Test Accuracy : ", acc)