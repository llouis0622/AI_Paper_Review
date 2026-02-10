import numpy as np


class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.1):
        self.nv = n_visible
        self.nh = n_hidden
        self.lr = lr
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sample_h(self, v):
        p = self.sigmoid(self.c + v @ self.W)
        return p, (p > np.random.rand(*p.shape)).astype(float)

    def sample_v(self, h):
        p = self.sigmoid(self.b + h @ self.W.T)
        return p, (p > np.random.rand(*p.shape)).astype(float)

    def train(self, X, epochs=10):
        for _ in range(epochs):
            for v0 in X:
                ph0, h0 = self.sample_h(v0)
                pv1, v1 = self.sample_v(h0)
                ph1, _ = self.sample_h(v1)

                self.W += self.lr * (np.outer(v0, ph0) - np.outer(v1, ph1))
                self.b += self.lr * (v0 - v1)
                self.c += self.lr * (ph0 - ph1)


if __name__ == "__main__":
    X = np.random.binomial(1, 0.5, size=(500, 20))
    rbm1 = RBM(20, 10)
    rbm1.train(X)

    H = np.array([rbm1.sample_h(x)[0] for x in X])
    rbm2 = RBM(10, 5)
    rbm2.train(H)
