from sklearn import datasets
from sklearn import linear_model
import numpy as np
import common
import MyLogisticReg

methods = {'MyLogisticReg2': MyLogisticReg.MyLogisticReg2(13),
           'LogisticRegression': linear_model.LogisticRegression(
               penalty='l2', solver='lbfgs',multi_class='multinomial', max_iter=5000)}

def hw3q3():
    # Constants
    k = 5

    # Datasets for q3
    q3_data = {}

    # Pre-processing Boston data set
    X, y = datasets.load_boston(return_X_y=True)
    q3_data['Boston50'] = {'X': X, 'y': np.where(y < np.percentile(y, 50), 0, 1)}
    q3_data['Boston25'] = {'X': X, 'y': np.where(y < np.percentile(y, 25), 0, 1)}

    combinations = [(method, dataset) for method in methods.keys() for dataset in q3_data.keys()]

    for method, dataset in combinations:
        print("Error rates for {} with {}:".format(method, dataset))
        common.my_cross_val(methods[method], q3_data[dataset]['X'], q3_data[dataset]['y'], k)
        print()

hw3q3()