import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


def linear_svm_demo():
    X, y = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        class_sep=1.0,
        flip_y=0.02,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for C in [0.1, 1.0, 10.0]:
        clf = LinearSVC(C=C, max_iter=30000, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"LinearSVC  C={C:<4}  acc={acc:.4f}")


def rbf_svm_demo():
    X, y = make_moons(n_samples=2500, noise=0.25, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for C in [0.3, 1.0, 3.0]:
        for gamma in [0.5, 1.0, 2.0]:
            clf = SVC(kernel="rbf", C=C, gamma=gamma)
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            n_sv = int(np.sum(clf.n_support_))
            print(f"RBF SVC  C={C:<4} gamma={gamma:<4}  acc={acc:.4f}  n_sv={n_sv}")


if __name__ == "__main__":
    linear_svm_demo()
    rbf_svm_demo()