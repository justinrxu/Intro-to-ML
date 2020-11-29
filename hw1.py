from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
import numpy as np

import common


# Methods
methods = {'LinearSVC': svm.LinearSVC(max_iter=2000), 'SVC': svm.SVC(gamma='scale', C=10),
           'LinearRegression': linear_model.LogisticRegression(penalty='l2', solver='lbfgs',
                                                               multi_class='multinomial', max_iter=5000)}


def q3i():
    # Constants
    k = 10

    # Data sets for q3i
    q3i_data = {}

    # Pre-processing Boston data set
    X, y = datasets.load_boston(return_X_y=True)
    q3i_data['Boston50'] = {'X': X, 'y': np.where(y < np.percentile(y, 50), 0, 1)}
    q3i_data['Boston25'] = {'X': X, 'y': np.where(y < np.percentile(y, 25), 0, 1)}

    # Loading Digits data set
    X, y = datasets.load_digits(return_X_y=True)
    q3i_data['Digits'] = {'X': X, 'y': y}

    combinations = [(method, dataset) for method in methods.keys() for dataset in q3i_data.keys()]

    # Executing all methods on all data sets
    for method, dataset in combinations:
        print("Error rates for {} with {}:".format(method, dataset))
        common.my_cross_val(methods[method], q3i_data[dataset]['X'], q3i_data[dataset]['y'], k)
        print()


def q3ii():
    # Constants
    pi = 0.75
    k = 10

    # Data sets for q3ii
    q3ii_data = {}

    # Pre-processing Boston data set
    X, y = datasets.load_boston(return_X_y=True)
    q3ii_data['Boston50'] = {'X': X, 'y': np.where(y < np.percentile(y, 50), 0, 1)}
    q3ii_data['Boston25'] = {'X': X, 'y': np.where(y < np.percentile(y, 25), 0, 1)}

    # Loading Digits data set
    X, y = datasets.load_digits(return_X_y=True)
    q3ii_data['Digits'] = {'X': X, 'y': y}

    combinations = [(method, dataset) for method in methods.keys() for dataset in q3ii_data.keys()]

    # Executing all methods on all data sets
    for method, dataset in combinations:
        print("Error rates for {} with {}:".format(method, dataset))
        common.my_train_test(methods[method], q3ii_data[dataset]['X'], q3ii_data[dataset]['y'], pi, k)
        print()


def q4():
    # Constants
    d = 32

    # Data sets for q4
    q4_data = {}
    X, y = datasets.load_digits(return_X_y=True)

    # Pre-processing Digits data set
    q4_data['X1'] = {'X': common.rand_proj(X, d), 'y': y}
    q4_data['X2'] = {'X': common.quad_proj(X, d), 'y': y}

    combinations = [(method, dataset) for method in methods.keys() for dataset in q4_data.keys()]

    # Executing all methods on all data sets
    for method, dataset in combinations:
        print("Error rates for {} with {}:".format(method, dataset))
        # Performing 10 fold cross validation
        common.my_cross_val(methods[method], q4_data[dataset]['X'], q4_data[dataset]['y'], 10)
        print()


q3i()
q3ii()
q4()
