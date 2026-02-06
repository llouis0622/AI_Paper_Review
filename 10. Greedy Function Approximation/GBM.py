import numpy as np


class DecisionStump:
    def fit(self, X, y):
        n, d = X.shape
        best_loss = float("inf")

        for j in range(d):
            values = np.unique(X[:, j])
            for t in values:
                left = X[:, j] <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue

                l_val = y[left].mean()
                r_val = y[right].mean()
                pred = np.where(left, l_val, r_val)
                loss = ((y - pred) ** 2).sum()

                if loss < best_loss:
                    best_loss = loss
                    self.feature = j
                    self.threshold = t
                    self.left_value = l_val
                    self.right_value = r_val

    def predict(self, X):
        return np.where(
            X[:, self.feature] <= self.threshold,
            self.left_value,
            self.right_value
        )


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, lr=0.1):
        self.n_estimators = n_estimators
        self.lr = lr
        self.models = []

    def fit(self, X, y):
        self.init_value = y.mean()
        pred = np.full_like(y, self.init_value, dtype=float)

        for _ in range(self.n_estimators):
            residual = y - pred
            stump = DecisionStump()
            stump.fit(X, residual)
            update = stump.predict(X)
            pred += self.lr * update
            self.models.append(stump)

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_value)
        for m in self.models:
            pred += self.lr * m.predict(X)
        return pred


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(200, 1)
    y = np.sin(6 * X[:, 0]) + np.random.randn(200) * 0.1

    model = GradientBoostingRegressor(n_estimators=100, lr=0.1)
    model.fit(X, y)
    pred = model.predict(X)
