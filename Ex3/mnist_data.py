import tensorflow as tf
from matplotlib import pyplot as plt
from plotnine import *
from sklearn.neighbors import KNeighborsClassifier
from models import *
from pandas import DataFrame
import timeit

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]
m_array = [50, 100, 300, 500]


def q12():
    zero_imgs = np.argwhere(y_train == 0).flatten()
    zero_idx = zero_imgs[:3]
    for i in zero_idx:
        plt.imshow(x_train[i])
        plt.show()
    one_imgs = np.argwhere(y_train == 1).flatten()
    one_idx = one_imgs[:3]
    for i in one_idx:
        plt.imshow(x_train[i])
        plt.show()


def draw_points(m):
    idx = np.random.choice(x_train.shape[0], m)
    X = x_train[idx]
    y = y_train[idx]
    return X, y


def rearrange_data(X):
    return np.reshape(X, (X.shape[0], 784))


def q14():
    log_reg = LogisticReg()
    svm = SVC(C=1, kernel='linear')
    dec_tree = DecisionT(max_depth=15)
    knn = KNeighborsClassifier()
    # init accuracies lists for every m
    log_reg_accuracies, svm_accuracies = [], []
    dec_t_accuracies, knn_accuracies = [], []
    log_reg_time, svm_time = [], []
    dec_t_time, knn_time = [], []
    for m in m_array:
        check_time = True
        log_reg_acc, svm_acc, dec_t_acc, knn_acc = 0, 0, 0, 0
        for i in range(50):
            train_pts, train_labels = draw_points(m)
            while np.abs(np.sum(y_train)) == m:
                train_pts, train_labels = draw_points(m)
            train_pts = rearrange_data(train_pts)
            # change 0 label to -1, 1 remains 1
            train_labels = np.where(train_labels == 1, train_labels,
                                    np.repeat(-1, train_labels.size))
            x_test_pts = rearrange_data(x_test)
            y_test_pts = np.where(y_test == 1, y_test,
                                  np.repeat(-1, y_test.size))

            # train and check times
            start_time = timeit.default_timer()
            log_reg.fit(train_pts, train_labels)
            end_time = timeit.default_timer()
            if check_time:
                log_reg_time.append(end_time - start_time)
            start_time = timeit.default_timer()
            svm.fit(train_pts, train_labels)
            end_time = timeit.default_timer()
            if check_time:
                svm_time.append(end_time - start_time)
            start_time = timeit.default_timer()
            dec_tree.fit(train_pts, train_labels)
            end_time = timeit.default_timer()
            if check_time:
                dec_t_time.append(end_time - start_time)
            start_time = timeit.default_timer()
            knn.fit(train_pts, train_labels)
            end_time = timeit.default_timer()
            if check_time:
                knn_time.append(end_time - start_time)
            check_time = False
            # accuracy for current iteration
            log_reg_acc += log_reg.score(x_test_pts, y_test_pts)['accuracy']
            svm_acc += svm.score(x_test_pts,
                                 y_test_pts)  # built in, only accuracy
            dec_t_acc += dec_tree.score(x_test_pts, y_test_pts)['accuracy']
            knn_acc += knn.score(x_test_pts,
                                 y_test_pts)  # built in, only accuracy

        # calculate mean accuracy for current m
        log_reg_accuracies.append(log_reg_acc / 50)
        svm_accuracies.append(svm_acc / 50)
        dec_t_accuracies.append(dec_t_acc / 50)
        knn_accuracies.append(knn_acc / 50)

    p = (ggplot() + geom_line(aes(x='m', y='y', color='legend'),
                              data=DataFrame(
                                  {'m': m_array, 'y': log_reg_accuracies,
                                   'legend': ['Logistic Regression'] * len(
                                       m_array)}), size=0.5) +
         geom_line(aes(x='m', y='y', color='legend'),
                   data=DataFrame({'m': m_array, 'y': svm_accuracies,
                                   'legend': ['SVM'] * len(
                                       m_array)}), size=0.5) +
         geom_line(aes(x='m', y='y', color='legend'),
                   data=DataFrame({'m': m_array, 'y': dec_t_accuracies,
                                   'legend': ['Decision Tree'] * len(
                                       m_array)}), size=0.5) +
         geom_line(aes(x='m', y='y', color='legend'),
                   data=DataFrame({'m': m_array, 'y': knn_accuracies,
                                   'legend': ['K-Nearest Neighbors'] * len(
                                       m_array)}), size=0.5) + ylab(
                'Accuracy') +
         ggtitle("Mean Accuracy as a function of m samples"))
    ggsave(p, "Q14.png", verbose=False)
    print("Logistic Regression Time: ", log_reg_time)
    print("SVM Time: ", svm_time)
    print("Decision Tree Time: ", dec_t_time)
    print("K-Nearest Neighbors Time: ", knn_time)


q12()
q14()
