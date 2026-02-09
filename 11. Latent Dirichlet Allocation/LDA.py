import numpy as np
from scipy.special import digamma


class VariationalLDA:
    def __init__(self, n_topics, alpha, beta):
        self.k = n_topics
        self.alpha = alpha
        self.beta = beta

    def infer_document(self, doc, max_iter=50):
        N = len(doc)
        gamma = self.alpha + N / self.k
        phi = np.ones((N, self.k)) / self.k

        for _ in range(max_iter):
            for n in range(N):
                for i in range(self.k):
                    phi[n, i] = self.beta[i, doc[n]] * np.exp(
                        digamma(gamma[i]) - digamma(np.sum(gamma))
                    )
                phi[n] /= np.sum(phi[n])
            gamma = self.alpha + np.sum(phi, axis=0)
        return gamma, phi


if __name__ == "__main__":
    vocab_size = 6
    topics = 2
    alpha = np.array([0.5, 0.5])
    beta = np.array([
        [0.4, 0.4, 0.1, 0.05, 0.03, 0.02],
        [0.05, 0.05, 0.2, 0.3, 0.2, 0.2]
    ])

    doc = [0, 1, 0, 2, 3, 3, 4]
    lda = VariationalLDA(topics, alpha, beta)
    gamma, phi = lda.infer_document(doc)
    print(gamma)
