import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# q9
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


# q10
def predict(X, w):
    """
    :param X: design matrix, a numpy array with p rows and n columns
    :param w: coefficients vector
    :return: predicted value by the model
    """
    # y_hat = (X^Transposed) * w
    return np.matmul(X.T, w)


# q11
def mse(y, w):
    """
    :param y: response vector, a numpy array
    :param w: prediction vector, a numpy array
    :return: MSE over the received samples
    """
    # np.linalg.norm calculates the L2 (Frobenius) norm by default
    # god bless numpy
    return ((np.linalg.norm(w - y)) ** 2) / len(w)


# q12, 13
def load_data(file_path):
    """
    :param file_path: csv file
    :return: 2 dataframes - the design matrix and prediction vector,
    after removing false values, irrelevant columns and adding intercept
    """
    df = pd.read_csv(file_path)
    df.dropna()
    df = df[(df['id'] != 0) & (df['date'] != 0) & (df['price'] > 0) &
            (df['sqft_living'] > 0) & (df['sqft_lot'] > 0) & (df['floors'] > 0)
            & (0 <= df['view']) & (df['view'] <= 4) & (1 <= df['condition'])
            & (df['condition'] <= 5) & (1 <= df['grade']) & (df['grade'] <= 13)
            & (df['sqft_above'] > 0) & (df['yr_built'] > 0) &
            (df['zipcode'] > 0) & (df['lat'] > 0) & (df['long'] < 0) &
            (df['sqft_living15'] > 0) & (df['sqft_lot15'] > 0)]
    zip_codes = pd.get_dummies(df['zipcode'])
    df = pd.concat([df, zip_codes], axis='columns')
    df1 = pd.DataFrame([[1] * len(df.columns)], columns=df.columns)
    df = pd.concat([df1, df], axis='rows').reset_index(drop=True)
    response_vector = df['price']
    df = df.drop(columns=['id', 'lat', 'long', 'price', 'zipcode', 'date'])

    return df, response_vector


def plot_singular_values(singular_values):
    """
    Plots the scree-plot of singular values in a descending order.
    :param singular_values: singular values numpy array
    """
    s_values = np.sort(singular_values)[::-1]
    plt.plot(np.arange(len(s_values)), s_values, '-o')
    plt.title("Singular Values Scree-plot")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.show()


def q_15():
    """
    Fits the house data and plots the singular values.
    """
    X, y = load_data("kc_house_data.csv")
    sigma = fit_linear_regression(X.T, y)[1]
    plot_singular_values(sigma)


def q_16():
    """
    Trains and test our modle and plots the MSE over the test set.
    """
    X, y = load_data("kc_house_data.csv")
    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test = train_test_split(X, test_size=0.25)
    y_train, y_test = train_test_split(y, test_size=0.25)
    train_mse = []
    for p in range(1, 101):
        # X,y have the same number of rows
        split_idx = int( (p / 100) * len(X_train))
        p_x_train = X_train[:split_idx]
        p_y_train = y_train[:split_idx]
        w = fit_linear_regression(p_x_train.T, p_y_train)[0]
        y_predict = predict(X_test.T, w)
        train_mse.append(mse(y_test, y_predict))
    plt.plot(np.arange(1, 101) / 100, train_mse, '-o')
    plt.xlabel("Percentage of training set")
    plt.ylabel("MSE")
    plt.title("MSE over the test set as a function of p%")
    plt.show()


def feature_evaluation(X, y):
    """
    :param X: design matirx
    :param y: prediction vector
    :return: Plots features and the prediction vector of house prices and
    prints the relevant Person correlation.
    """
    features = np.array(X.columns[:14])
    for i, feature in enumerate(features):
        pearson_correlation = np.cov(X[feature], y)[0,1] / (
                    np.std(X[feature]) * np.std(y))
        plt.plot(X[feature], y, 'o')
        plt.xlabel("Feature: {}".format(feature))
        plt.ylabel("Prediction vector: house prices")
        plt.title(
            "Scatter plot of {} and house prices with \n Pearson correlation of: {}".format(
                feature, np.round(pearson_correlation, 2)))
        plt.show()
