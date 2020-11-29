from sklearn import model_selection
from sklearn.metrics import accuracy_score
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def error_rate(pred, obs):
    return 1-accuracy_score(obs, pred)


# HW 1 methods

def kfsplit(X, k):
    fold_indices = np.array_split(np.arange(X.shape[0], dtype=int), k)
    train_test_indices = []
    for i in range(k):
        train_indices = np.array([])
        test_indices = []
        for fold in range(len(fold_indices)):
            if fold == i:
                test_indices = fold_indices[fold]
            else:
                train_indices = np.concatenate((train_indices.astype(int), fold_indices[fold]))
        train_test_indices.append((train_indices, test_indices))
    return train_test_indices


def my_cross_val(method, X, y, k, errorfunc=error_rate):
    error_rates = []
    for train_index, test_index in kfsplit(X, k):
        method.fit(X[train_index], y[train_index])
        error_rates.append(errorfunc(method.predict(X[test_index]), y[test_index]))
        print("Fold {}: {}".format(len(error_rates), error_rates[len(error_rates) - 1]))
    print("Mean: {}".format(np.mean(np.array(error_rates))))
    print("Standard Deviation: {}".format(np.std(np.array(error_rates))))


def my_train_test(method, X, y, pi, k):
    error_rates = []
    for i in range(k):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=pi)
        method.fit(X_train, y_train)
        error_rates.append(error_rate(method.predict(X_test), y_test))
        print("Fold {}: {}".format(len(error_rates), error_rates[len(error_rates) - 1]))
    print("Mean: {}".format(np.mean(np.array(error_rates))))
    print("Standard Deviation: {}".format(np.std(np.array(error_rates))))


# HW 2 Methods

def rand_proj(X, d):
    return np.matmul(X, np.random.normal(size=(X.shape[1], d)))


def quad_proj(X):
    X_square = np.square(X)
    # All index combinations found in upper triangular matrix with right offset 1
    feature_cross_combs = np.triu_indices(X.shape[1], 1)
    # 'cross multiply' with combinations as indices
    X_feature_cross = np.multiply(X[:, [feature_cross_combs[0]]][:, 0, :], X[:, [feature_cross_combs[1]]][:, 0, :])
    return np.concatenate((X, X_square, X_feature_cross), axis=1)