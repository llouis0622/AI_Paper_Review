import numpy as np


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime_from_y(y):
    return y * (1.0 - y)


def mse(y, t):
    return 0.5 * np.mean((y - t) ** 2)


def add_bias(X):
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)


class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.5, size=(in_dim + 1, hidden_dim))
        self.W2 = rng.normal(0.0, 0.5, size=(hidden_dim + 1, out_dim))

    def forward(self, X):
        Xb = add_bias(X)
        z1 = Xb @ self.W1
        h = sigmoid(z1)
        hb = add_bias(h)
        z2 = hb @ self.W2
        y = sigmoid(z2)
        cache = (Xb, h, hb, y)
        return y, cache

    def grads(self, X, T):
        Y, cache = self.forward(X)
        Xb, h, hb, y = cache

        dE_dy = (y - T) / X.shape[0]
        dy_dz2 = sigmoid_prime_from_y(y)
        delta2 = dE_dy * dy_dz2

        gW2 = hb.T @ delta2

        delta1_full = (delta2 @ self.W2.T)
        delta1 = delta1_full[:, :-1]
        dh_dz1 = sigmoid_prime_from_y(h)
        delta1 = delta1 * dh_dz1

        gW1 = Xb.T @ delta1
        loss = mse(y, T)
        return loss, gW1, gW2

    def step(self, X, T, lr=0.5, momentum=0.0, state=None):
        loss, gW1, gW2 = self.grads(X, T)

        if state is None:
            state = {"vW1": np.zeros_like(self.W1), "vW2": np.zeros_like(self.W2)}

        state["vW1"] = momentum * state["vW1"] - lr * gW1
        state["vW2"] = momentum * state["vW2"] - lr * gW2

        self.W1 += state["vW1"]
        self.W2 += state["vW2"]
        return loss, state


def grad_check(model, X, T, eps=1e-5, samples=25, seed=0):
    rng = np.random.default_rng(seed)
    loss, gW1, gW2 = model.grads(X, T)

    def loss_only():
        y, _ = model.forward(X)
        return mse(y, T)

    diffs = []

    idxs = [(0, "W1", rng.integers(model.W1.shape[0]), rng.integers(model.W1.shape[1])) for _ in range(samples // 2)]
    idxs += [(0, "W2", rng.integers(model.W2.shape[0]), rng.integers(model.W2.shape[1])) for _ in range(samples - len(idxs))]

    for _, which, i, j in idxs:
        if which == "W1":
            orig = model.W1[i, j]
            model.W1[i, j] = orig + eps
            lp = loss_only()
            model.W1[i, j] = orig - eps
            lm = loss_only()
            model.W1[i, j] = orig
            num = (lp - lm) / (2.0 * eps)
            ana = gW1[i, j]
        else:
            orig = model.W2[i, j]
            model.W2[i, j] = orig + eps
            lp = loss_only()
            model.W2[i, j] = orig - eps
            lm = loss_only()
            model.W2[i, j] = orig
            num = (lp - lm) / (2.0 * eps)
            ana = gW2[i, j]

        denom = max(1e-12, abs(num) + abs(ana))
        rel = abs(num - ana) / denom
        diffs.append((which, i, j, num, ana, rel))

    diffs.sort(key=lambda x: -x[-1])
    return loss, diffs[:10]


def train_xor():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float64)
    T = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float64)

    model = MLP(in_dim=2, hidden_dim=2, out_dim=1, seed=7)

    loss0, top = grad_check(model, X, T, eps=1e-5, samples=30, seed=1)
    print("grad_check loss : ", float(loss0))
    print("top relative diffs : ")
    for which, i, j, num, ana, rel in top:
        print(f"  {which}[{i},{j}]  num={num:.6e}  ana={ana:.6e}  rel={rel:.3e}")

    state = None
    lr = 1.0
    momentum = 0.9
    steps = 8000

    for k in range(1, steps + 1):
        loss, state = model.step(X, T, lr=lr, momentum=momentum, state=state)
        if k % 500 == 0 or k == 1:
            y, _ = model.forward(X)
            preds = (y >= 0.5).astype(int)
            acc = float(np.mean(preds == T.astype(int)))
            print(f"step={k:5d}  loss={loss:.6f}  acc={acc:.3f}")

    y, _ = model.forward(X)
    print("\nfinal outputs : ")
    for x, yy in zip(X, y):
        print(f"{x.tolist()} -> {float(yy[0]):.6f}")


if __name__ == "__main__":
    train_xor()