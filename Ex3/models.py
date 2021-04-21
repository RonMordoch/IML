import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def score_dict(X, y, y_hat):
    pos = np.sum(y == 1)
    neg = np.sum(y == -1)
    false_pos = np.sum(np.logical_and(y_hat == 1, y != y_hat))
    true_pos = np.sum(np.logical_and(y_hat == 1, y == y_hat))

    num_samples = X.shape[1]  # number of columns, m'
    error = np.sum(y != y_hat) / y.size
    accuracy = np.sum(y == y_hat) / y.size
    false_pos_rate = false_pos / neg
    true_pos_rate = true_pos / pos
    precision = true_pos / np.sum(y_hat == 1)

    return {'num_samples': num_samples, 'error': error,
            'accuracy': accuracy, 'FPR': false_pos_rate,
            'TPR': true_pos_rate, 'precision': precision,
            'recall': true_pos_rate}


class Perceptron:

    def __init__(self):
        self.model = None

    def fit(self, X, y):  # Perceptron algorithm
        # X is in R(dxm), w in R(d)
        w = np.repeat(0.0, X.shape[0] + 1)  # X.shape[0] = d, extra 1 for b
        # inserts 1s to X, save training set
        X = np.vstack((np.repeat(1, X.shape[1]), X))
        while True:
            options = y * np.matmul(w.T, X)
            indexes = np.argwhere(options <= 0)
            # all indices satisfying the if condition
            if indexes.size == 0:  # no such i exists, end the algorithm
                self.model = w
                break
            i = indexes[0]  # choose an index
            w += (y[i] * X.T[i]).flatten()
            # regular multiplication creates a nested array

    def predict(self, X):
        X_ones = np.vstack((np.repeat(1, X.shape[1]), X))
        predicts = np.sign(np.matmul(self.model, X_ones))
        predicts[predicts == 0] = 1  # classify 0 as 1, be positive :)
        return predicts

    def score(self, X, y):
        y_hat = self.predict(X)
        return score_dict(X, y, y_hat)


class LDA:

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        x_labeled_pos = X[:, np.argwhere(y == 1).flatten()]
        x_labeled_neg = X[:, np.argwhere(y == -1).flatten()]
        # mean value of sum of all the columns classified as 1 or -1
        mu_pos = np.sum(x_labeled_pos, axis=1) / x_labeled_pos.shape[1]
        mu_neg = np.sum(x_labeled_neg, axis=1) / x_labeled_neg.shape[1]
        sigma_inv = np.linalg.inv(np.cov(X))

        # as described in UML book
        w = (mu_pos - mu_neg).T.dot(sigma_inv)
        b = mu_neg.T.dot(sigma_inv.dot(mu_neg)) - mu_neg.T.dot(
            sigma_inv.dot(mu_neg))
        b *= 0.5
        self.model = np.hstack((b, w))

    def predict(self, X):
        X_ones = np.vstack((np.repeat(1, X.shape[1]), X))
        predicts = np.sign(np.matmul(self.model, X_ones))
        predicts[predicts == 0] = 1  # classify 0 as 1, be positive :)
        return predicts

    def score(self, X, y):
        y_hat = self.predict(X)
        return score_dict(X, y, y_hat)


class SVM:

    def __init__(self):
        self.clf = SVC(C=1e10, kernel='linear')
        self.model = None

    def fit(self, X, y):
        self.clf.fit(X.T, y)
        intercept = self.clf.intercept_
        coef = self.clf.coef_[0]
        self.model = np.array([intercept, coef[0], coef[1]])

    def predict(self, X):
        return self.clf.predict(X.T)

    def score(self, X, y):
        y_hat = self.predict(X)
        return score_dict(X, y, y_hat)


class LogisticReg:

    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_hat = self.predict(X)
        return score_dict(X, y, y_hat)


class DecisionT:

    def __init__(self, max_depth):
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_hat = self.predict(X)
        return score_dict(X, y, y_hat)
