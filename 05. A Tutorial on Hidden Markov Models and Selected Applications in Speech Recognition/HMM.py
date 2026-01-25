import numpy as np


def _normalize(v, axis=None, eps=1e-12):
    s = np.sum(v, axis=axis, keepdims=True)
    s = np.maximum(s, eps)
    return v / s


def _logsumexp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


class DiscreteHMM:
    def __init__(self, n_states, n_obs, seed=0):
        self.n_states = int(n_states)
        self.n_obs = int(n_obs)
        rng = np.random.default_rng(seed)
        self.pi = _normalize(rng.random(self.n_states), axis=0)
        self.A = _normalize(rng.random((self.n_states, self.n_states)), axis=1)
        self.B = _normalize(rng.random((self.n_states, self.n_obs)), axis=1)

    def _forward_scaled(self, obs):
        obs = np.asarray(obs, dtype=np.int64)
        T = obs.shape[0]
        N = self.n_states

        alpha = np.zeros((T, N), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        alpha[0] = self.pi * self.B[:, obs[0]]
        c[0] = np.sum(alpha[0]) + 1e-300
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = np.sum(alpha[t]) + 1e-300
            alpha[t] /= c[t]

        logp = -np.sum(np.log(c))
        return alpha, c, logp

    def _backward_scaled(self, obs, c):
        obs = np.asarray(obs, dtype=np.int64)
        T = obs.shape[0]
        N = self.n_states

        beta = np.zeros((T, N), dtype=np.float64)
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            beta[t] /= (c[t + 1] + 1e-300)

        return beta

    def loglik(self, obs):
        _, _, logp = self._forward_scaled(obs)
        return float(logp)

    def forward_backward(self, obs):
        alpha, c, logp = self._forward_scaled(obs)
        beta = self._backward_scaled(obs, c)

        gamma = alpha * beta
        gamma = _normalize(gamma, axis=1)

        obs = np.asarray(obs, dtype=np.int64)
        T = obs.shape[0]
        xi = np.zeros((T - 1, self.n_states, self.n_states), dtype=np.float64)

        for t in range(T - 1):
            b = self.B[:, obs[t + 1]] * beta[t + 1]
            num = (alpha[t][:, None] * self.A) * b[None, :]
            den = np.sum(num) + 1e-300
            xi[t] = num / den

        return gamma, xi, float(logp)

    def viterbi(self, obs):
        obs = np.asarray(obs, dtype=np.int64)
        T = obs.shape[0]
        N = self.n_states

        log_pi = np.log(self.pi + 1e-300)
        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)

        dp = np.zeros((T, N), dtype=np.float64)
        back = np.zeros((T, N), dtype=np.int64)

        dp[0] = log_pi + log_B[:, obs[0]]
        back[0] = 0

        for t in range(1, T):
            scores = dp[t - 1][:, None] + log_A
            back[t] = np.argmax(scores, axis=0)
            dp[t] = scores[back[t], np.arange(N)] + log_B[:, obs[t]]

        path = np.zeros(T, dtype=np.int64)
        path[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]

        best_logp = float(np.max(dp[T - 1]))
        return path.tolist(), best_logp

    def baum_welch(self, sequences, n_iter=30, tol=1e-6, smoothing=1e-9, verbose=True):
        seqs = [np.asarray(s, dtype=np.int64) for s in sequences]
        last = None

        for it in range(1, n_iter + 1):
            pi_acc = np.zeros(self.n_states, dtype=np.float64)
            A_num = np.zeros((self.n_states, self.n_states), dtype=np.float64)
            A_den = np.zeros(self.n_states, dtype=np.float64)
            B_num = np.zeros((self.n_states, self.n_obs), dtype=np.float64)
            B_den = np.zeros(self.n_states, dtype=np.float64)

            total_ll = 0.0

            for obs in seqs:
                gamma, xi, ll = self.forward_backward(obs)
                total_ll += ll

                pi_acc += gamma[0]
                A_num += np.sum(xi, axis=0)
                A_den += np.sum(gamma[:-1], axis=0)

                for t, o in enumerate(obs):
                    B_num[:, o] += gamma[t]
                B_den += np.sum(gamma, axis=0)

            self.pi = _normalize(pi_acc + smoothing, axis=0)
            self.A = _normalize(A_num + smoothing, axis=1)
            self.B = _normalize(B_num + smoothing, axis=1)

            if verbose:
                print(f"iter={it:02d}  total_loglik={total_ll:.6f}")

            if last is not None:
                if abs(total_ll - last) <= tol * (1.0 + abs(last)):
                    break
            last = total_ll

        return float(last if last is not None else total_ll)


def sample_hmm(pi, A, B, T, seed=0):
    rng = np.random.default_rng(seed)
    pi = np.asarray(pi, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    N = pi.shape[0]
    M = B.shape[1]

    states = np.zeros(T, dtype=np.int64)
    obs = np.zeros(T, dtype=np.int64)

    states[0] = rng.choice(N, p=pi)
    obs[0] = rng.choice(M, p=B[states[0]])

    for t in range(1, T):
        states[t] = rng.choice(N, p=A[states[t - 1]])
        obs[t] = rng.choice(M, p=B[states[t]])

    return states.tolist(), obs.tolist()


def demo():
    pi_true = np.array([0.6, 0.4], dtype=np.float64)
    A_true = np.array([[0.7, 0.3],
                       [0.4, 0.6]], dtype=np.float64)
    B_true = np.array([[0.5, 0.4, 0.1],
                       [0.1, 0.3, 0.6]], dtype=np.float64)

    seqs = []
    for k in range(50):
        _, o = sample_hmm(pi_true, A_true, B_true, T=30, seed=100 + k)
        seqs.append(o)

    model = DiscreteHMM(n_states=2, n_obs=3, seed=7)
    model.baum_welch(seqs, n_iter=40, tol=1e-7, verbose=True)

    test_states, test_obs = sample_hmm(pi_true, A_true, B_true, T=25, seed=999)
    path, best_logp = model.viterbi(test_obs)
    ll = model.loglik(test_obs)

    print("\nlearned pi : \n", model.pi)
    print("learned A : \n", model.A)
    print("learned B : \n", model.B)
    print("\nViterbi path(first 20) : ", path[:20])
    print("Viterbi best logp : ", best_logp)
    print("Forward loglik : ", ll)


if __name__ == "__main__":
    demo()