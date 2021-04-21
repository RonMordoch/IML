"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
from ex4_tools import *
from matplotlib import pyplot as plt


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.D = None

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = np.array([1 / m] * m)
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            y_predicted = self.h[t].predict(X)
            epsilon_t = np.sum(D * (y != y_predicted))
            self.w[t] = 0.5 * np.log(1 / epsilon_t - 1)
            numerator = D * np.exp(-self.w[t] * y * y_predicted)
            denominator = np.sum(numerator)
            D = numerator / denominator
            self.D = D
        return self.w

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        prediction = np.zeros(X.shape[0])
        for t in range(max_t):
            prediction += self.w[t] * self.h[t].predict(X)
        return np.sign(prediction)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        predictions = self.predict(X, max_t)
        return np.sum(np.where(y != predictions,
                               np.ones(len(y)),
                               np.zeros(len(y)))) / len(y)

    def get_D(self):
        return self.D


def run_adaboost(noise_ratio):
    # Question 10
    X_train, y_train = generate_data(num_samples=5000, noise_ratio=noise_ratio)
    adaboost = AdaBoost(WL=DecisionStump,
                        T=500)  # stump initiliazation in train
    X_test, y_test = generate_data(num_samples=200, noise_ratio=noise_ratio)
    adaboost.train(X_train, y_train)
    train_errors, test_errors = [], []
    t_vals = np.arange(1, 501)
    for t in t_vals:
        train_errors.append(adaboost.error(X_train, y_train, t))
        test_errors.append(adaboost.error(X_test, y_test, t))
    plt.plot(t_vals, train_errors, label='Train Error')
    plt.plot(t_vals, test_errors, label='Test Error')
    plt.xlabel("T")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Train/Test Error as function of T with noise ratio={}".format(
        noise_ratio))
    plt.show()
    # Question 11
    t_vals = [5, 10, 50, 100, 200, 500]  # total 6 graphs, nrows=2, ncols=3
    for i in range(len(t_vals)):
        plt.subplot(2, 3, i + 1)
        decision_boundaries(classifier=adaboost, X=X_test, y=y_test,
                            num_classifiers=t_vals[i])
        plt.title("T ={}, noise ratio={}".format(t_vals[i], noise_ratio))
    plt.show()

    # Question 12
    test_error_min_t = np.argmin(test_errors)
    decision_boundaries(classifier=adaboost, X=X_train, y=y_train,
                        num_classifiers=test_error_min_t)
    plt.title(
        "Decision Boundary: T={} minimizing test error={}, noise ratio={}".format(
            test_error_min_t,
            test_errors[test_error_min_t],
            noise_ratio))
    plt.show()

    # Question 13
    T = 499
    D = adaboost.get_D()
    decision_boundaries(classifier=adaboost, X=X_train, y=y_train,
                        num_classifiers=T, weights=D)
    plt.title(
        "Weights of training set with noise ratio={}".format(noise_ratio))
    plt.show()
    D = D / np.max(D) * 10
    decision_boundaries(classifier=adaboost, X=X_train, y=y_train,
                        num_classifiers=T, weights=D)
    plt.title(
        "Normalized weights of training set with noise ratio={}".format(
            noise_ratio))
    plt.show()


def q14():
    for noise_ratio in [0, 0.01, 0.4]:
        run_adaboost(noise_ratio)


q14()
