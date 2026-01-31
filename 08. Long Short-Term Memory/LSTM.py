import numpy as np


def sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


class LSTMCell:
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = rng.normal(0.0, 0.1, (input_dim + hidden_dim, hidden_dim))
        self.W_f = rng.normal(0.0, 0.1, (input_dim + hidden_dim, hidden_dim))
        self.W_o = rng.normal(0.0, 0.1, (input_dim + hidden_dim, hidden_dim))
        self.W_g = rng.normal(0.0, 0.1, (input_dim + hidden_dim, hidden_dim))

        self.b_i = np.zeros(hidden_dim)
        self.b_f = np.ones(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        self.b_g = np.zeros(hidden_dim)

    def forward(self, x, h_prev, c_prev):
        concat = np.concatenate([x, h_prev])

        i = sigmoid(concat @ self.W_i + self.b_i)
        f = sigmoid(concat @ self.W_f + self.b_f)
        o = sigmoid(concat @ self.W_o + self.b_o)
        g = np.tanh(concat @ self.W_g + self.b_g)

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        cache = (x, h_prev, c_prev, concat, i, f, o, g, c)
        return h, c, cache

    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, concat, i, f, o, g, c = cache

        tanh_c = np.tanh(c)
        do = dh * tanh_c
        d_tanh_c = dh * o
        dc_total = dc + d_tanh_c * (1.0 - tanh_c ** 2)

        df = dc_total * c_prev
        di = dc_total * g
        dg = dc_total * i
        dc_prev = dc_total * f

        di_in = di * i * (1.0 - i)
        df_in = df * f * (1.0 - f)
        do_in = do * o * (1.0 - o)
        dg_in = dg * (1.0 - g ** 2)

        dconcat = di_in @ self.W_i.T + df_in @ self.W_f.T + do_in @ self.W_o.T + dg_in @ self.W_g.T
        dx = dconcat[:self.input_dim]
        dh_prev = dconcat[self.input_dim:]

        dW_i = np.outer(concat, di_in)
        dW_f = np.outer(concat, df_in)
        dW_o = np.outer(concat, do_in)
        dW_g = np.outer(concat, dg_in)

        db_i = di_in
        db_f = df_in
        db_o = do_in
        db_g = dg_in

        grads = (dW_i, dW_f, dW_o, dW_g, db_i, db_f, db_o, db_g)
        return dx, dh_prev, dc_prev, grads


class LSTMSequenceModel:
    def __init__(self, input_dim, hidden_dim, seed=0):
        self.cell = LSTMCell(input_dim, hidden_dim, seed=seed)
        rng = np.random.default_rng(seed + 1)
        self.Wy = rng.normal(0.0, 0.1, (hidden_dim, 1))
        self.by = np.zeros(1)

    def forward_sequence(self, X):
        T = X.shape[0]
        h = np.zeros(self.cell.hidden_dim)
        c = np.zeros(self.cell.hidden_dim)

        caches = []
        hs = np.zeros((T, self.cell.hidden_dim))
        ys = np.zeros((T, 1))

        for t in range(T):
            h, c, cache = self.cell.forward(X[t], h, c)
            y = h @ self.Wy + self.by
            hs[t] = h
            ys[t] = y
            caches.append((cache, h.copy(), c.copy()))

        return ys, hs, caches

    def loss_mse(self, Y, Tgt):
        return 0.5 * np.mean((Y - Tgt) ** 2)

    def train_copy_task(self, steps=3000, T=80, delay=50, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)

        def make_batch():
            X = np.zeros((T, 1))
            X[0, 0] = 1.0
            target = np.zeros((T, 1))
            target[delay, 0] = 1.0
            X += rng.normal(0.0, 0.02, X.shape)
            return X, target

        for step in range(1, steps + 1):
            X, target = make_batch()
            Y, hs, caches = self.forward_sequence(X)
            loss = self.loss_mse(Y, target)

            dY = (Y - target) / T
            dWy = hs.reshape(-1, self.cell.hidden_dim).T @ dY
            dby = np.sum(dY, axis=0)

            dh_next = np.zeros(self.cell.hidden_dim)
            dc_next = np.zeros(self.cell.hidden_dim)

            dW_i = np.zeros_like(self.cell.W_i)
            dW_f = np.zeros_like(self.cell.W_f)
            dW_o = np.zeros_like(self.cell.W_o)
            dW_g = np.zeros_like(self.cell.W_g)

            db_i = np.zeros_like(self.cell.b_i)
            db_f = np.zeros_like(self.cell.b_f)
            db_o = np.zeros_like(self.cell.b_o)
            db_g = np.zeros_like(self.cell.b_g)

            for t in range(T - 1, -1, -1):
                dh = (dY[t, 0] * self.Wy[:, 0]) + dh_next
                dx, dh_next, dc_next, grads = self.cell.backward(dh, dc_next, caches[t][0])
                gWi, gWf, gWo, gWg, gbi, gbf, gbo, gbg = grads

                dW_i += gWi
                dW_f += gWf
                dW_o += gWo
                dW_g += gWg

                db_i += gbi
                db_f += gbf
                db_o += gbo
                db_g += gbg

            for arr in [dW_i, dW_f, dW_o, dW_g, dWy, db_i, db_f, db_o, db_g, dby]:
                np.clip(arr, -1.0, 1.0, out=arr)

            self.cell.W_i -= lr * dW_i
            self.cell.W_f -= lr * dW_f
            self.cell.W_o -= lr * dW_o
            self.cell.W_g -= lr * dW_g

            self.cell.b_i -= lr * db_i
            self.cell.b_f -= lr * db_f
            self.cell.b_o -= lr * db_o
            self.cell.b_g -= lr * db_g

            self.Wy -= lr * dWy
            self.by -= lr * dby

            if step % 250 == 0 or step == 1:
                preds = (Y > 0.5).astype(int)
                acc = float(np.mean(preds == target.astype(int)))
                print(f"step={step:4d}  loss={loss:.6f}  acc={acc:.3f}  y@delay={float(Y[delay, 0]):.4f}")

        X, target = make_batch()
        Y, _, _ = self.forward_sequence(X)
        print("\nfinal check:")
        print("target delay index : ", delay)
        print("prediction at delay : ", float(Y[delay, 0]))
        print("max prediction index : ", int(np.argmax(Y)))


if __name__ == "__main__":
    model = LSTMSequenceModel(input_dim=1, hidden_dim=16, seed=7)
    model.train_copy_task(steps=2500, T=80, delay=50, lr=0.03, seed=1)