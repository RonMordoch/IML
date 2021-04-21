from models import *
from plotnine import *
from pandas import DataFrame

m_array = [5, 10, 15, 25, 70]


def draw_points(m):
    X = np.random.multivariate_normal([0, 0], np.identity(2), m).T
    y = np.sign(np.matmul([0.3, -0.5], X) + 0.1)
    y[y == 0] = 1
    return X, y


def q9():
    for m in m_array:
        X, y = draw_points(m)
        # svm fit fails if all labels are 1 or all labels are -1
        while np.abs(np.sum(y)) == m:
            X, y = draw_points(m)
        pos_pts = X.T[np.argwhere(y == 1).flatten()]
        neg_pts = X.T[np.argwhere(y == - 1).flatten()]
        xx = np.linspace(-5, 5, m)
        # true hypothesis
        true_hyperplane = 0.6 * xx + 0.2

        # get the Perceptron hyperplane
        perceptron = Perceptron()
        perceptron.fit(X, y)
        perceptron_hyperplane = (perceptron.model[0] / -perceptron.model[
            2]) + xx * (perceptron.model[1] / - perceptron.model[2])

        svm = SVM()
        svm.fit(X, y)
        # get the SVM hyperplane
        svm_hyperplane = (svm.model[0] / -svm.model[2]) + xx * (
                svm.model[1] / -svm.model[2])

        p = (ggplot() + geom_line(aes(x='x', y='y', color='legend'),
                                  data=DataFrame({'x': xx, 'y': svm_hyperplane,
                                                  'legend': ['SVM'] * len(
                                                      xx)}), size=0.5) +
             geom_line(aes(x='x', y='y', color='legend'),
                       data=DataFrame({'x': xx, 'y': true_hyperplane,
                                       'legend': ['True Hypothesis'] * len(
                                           xx)}), size=0.5) +
             geom_line(aes(x='x', y='y', color='legend'),
                       data=DataFrame({'x': xx, 'y': perceptron_hyperplane,
                                       'legend': ['Perceptron'] * len(
                                           xx)}), size=0.5) +
             geom_point(aes(x='x', y='y', color='legend'), data=DataFrame(
                 {'x': pos_pts[:, 0], 'y': pos_pts[:, 1],
                  'legend': ['Positive Label'] * len(pos_pts[:, 1])}),
                        size=2) +
             geom_point(aes(x='x', y='y', color='legend'), data=DataFrame(
                 {'x': neg_pts[:, 0], 'y': neg_pts[:, 1],
                  'legend': ['Negative Label'] * len(neg_pts[:, 1])}),
                        size=2)) + ggtitle(
            "%s Samples from 2D-Gaussian Distribution Classifications" % str(
                m))
        ggsave(p, "Q9_%s.png" % str(m), verbose=False)


def q10():
    perceptron = Perceptron()
    svm = SVM()
    lda = LDA()
    # init accuracies lists for every m
    perceptron_accuracies, svm_accuracies, lda_accuracies = [], [], []
    for m in m_array:
        perceptron_acc, svm_acc, lda_acc = 0, 0, 0
        for i in range(500):
            # gets valid test, train points
            X_train, y_train = draw_points(m)
            while np.abs(np.sum(y_train)) == m:
                X_train, y_train = draw_points(m)
            X_test, y_test = draw_points(10000)
            while np.abs(np.sum(y_test)) == m:
                X_test, y_test = draw_points(m)

            # train models
            perceptron.fit(X_train, y_train)
            svm.fit(X_train, y_train)
            lda.fit(X_train, y_train)
            # accuracy for current iteration
            perceptron_acc += perceptron.score(X_test, y_test)['accuracy']
            svm_acc += svm.score(X_test, y_test)['accuracy']
            lda_acc += lda.score(X_test, y_test)['accuracy']

        # calculate mean accuracy for current m
        perceptron_accuracies.append(perceptron_acc / 500)
        svm_accuracies.append(svm_acc / 500)
        lda_accuracies.append(lda_acc / 500)

    p = (ggplot() + geom_line(aes(x='m', y='y', color='legend'),
                              data=DataFrame(
                                  {'m': m_array, 'y': perceptron_accuracies,
                                   'legend': ['Perceptron'] * len(
                                       m_array)}), size=0.5) +
         geom_line(aes(x='m', y='y', color='legend'),
                   data=DataFrame({'m': m_array, 'y': svm_accuracies,
                                   'legend': ['SVM'] * len(
                                       m_array)}), size=0.5) +
         geom_line(aes(x='m', y='y', color='legend'),
                   data=DataFrame({'m': m_array, 'y': lda_accuracies,
                                   'legend': ['LDA'] * len(
                                       m_array)}), size=0.5) + ylab(
                'Accuracy') +
         ggtitle("Mean Accuracy as a function of m samples"))
    ggsave(p, "Q10.png", verbose=False)


q9()
q10()
