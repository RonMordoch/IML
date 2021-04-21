import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):
    """
    :param X: design matrix, a numpy array with p rows and n columns
    ( assumes X already has rows of 1's inserted to is)
    :param y: response vector, a numpy array with n rows
    :return: w , the coefficients vector; sigma, the singular values of X;
    """
    w = np.matmul(np.linalg.pinv(X.T), y)
    # calculate the SVD of X in order to gain sigma, dont calculate UV
    sigma = np.linalg.svd(X, compute_uv=False)
    return w, sigma


def predict(X, w):
    """
    :param X: design matrix, a numpy array with p rows and n columns
    :param w: coefficients vector
    :return: predicted value by the model
    """
    # y_hat = (X^Transposed) * w
    return np.matmul(X.T, w)


def load_data():
    """
    Reads the CSV files and adds the log_detected column.
    """
    df = pd.read_csv("covid19_israel.csv")
    df['log_detected'] = np.log(df['detected'])
    return df


# q20,21
def q_20_21():
    # convert df to numpy, add intercept and get the specified columns
    df = load_data().to_numpy()
    day_nums = df.T[0]
    intercept = np.repeat(1, day_nums.size)
    data = np.append([intercept], [day_nums], axis=0)
    data = data.astype('float')  # avoid casting issues
    log_detected = df.T[3]
    w = fit_linear_regression(data, log_detected)[0]
    # first figure - log detected cases as a function of day numbers
    fig1, ax1 = plt.subplots()
    ax1.plot(day_nums, log_detected, 'o')
    ax1.set(xlabel="Days numbers", ylabel="Log Detected",
            title="Log of detected cases as a function of day numbers")
    ax1.plot(day_nums, predict(data, w), '-')
    plt.show()
    # second figure - detected cases as a function of day numbers
    detected = df.T[2]
    fig2, ax2 = plt.subplots()
    ax2.plot(day_nums, detected, 'o')
    ax2.set(xlabel="Days numbers", ylabel="Detected case",
            title="Detected cases as a function of day numbers")
    w = w.astype('float')  # avoid casting issues
    ax2.plot(day_nums, np.exp(predict(data, w)), '-')
    plt.show()
