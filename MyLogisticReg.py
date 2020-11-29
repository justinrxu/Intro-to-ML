import numpy as np
import math
import random


def myshuffle(list1, list2):
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    return zip(*temp)


class MyLogisticReg2:
    def __init__(self, d):
        self.d = d
        self.w = [0.02 * (np.random.random() - 0.5) for j in range(self.d)]
        self.w0 = 0.02 * (np.random.random() - 0.5)
        self.max_iter = 500
        self.yeta = 0.00002
        return

    def fit(self, X_in, y_in):
        X = np.copy(X_in)
        y = np.copy(y_in)

        counter_iter_without_improvement = 0
        max_iter_without_improvement = 50

        batchsize = 4
        number_of_batches = int(len(X) / batchsize)

        min_error = 1.e15

        for iteration in range(self.max_iter):
            X, y = myshuffle(X, y)
            error = 0

            for i in range(number_of_batches):
                for k in range(batchsize):
                    WtX = np.dot(self.w, X[i * batchsize + k]) + self.w0
                    yi = 1 / (1 + np.exp(-1 * WtX))
                    for j in range(self.d):
                        self.w[j] = self.w[j] + self.yeta * (y[i * batchsize + k] - yi) * X[i * batchsize + k][
                            j] / batchsize
                    self.w0 = self.w0 + self.yeta * (y[i * batchsize + k] - yi) / batchsize

                    error = error + (y[i * batchsize + k] - yi) * (y[i * batchsize + k] - yi)
            error = math.sqrt(error / (number_of_batches * batchsize))
            if error < min_error:
                min_error = error
                counter_iter_without_improvement = 0
            elif error > min_error:
                counter_iter_without_improvement = counter_iter_without_improvement + 1

            if counter_iter_without_improvement > max_iter_without_improvement:
                return
        return

    def predict(self, X):
        y = [0 for i in range(len(X))]
        for i in range(len(X)):
            WtX = np.dot(self.w, X[i]) + self.w0
            if WtX > 0:
                y[i] = 1
            else:
                y[i] = 0
        return y

    def get_params(self, deep=True):
        return {"d": self.d}