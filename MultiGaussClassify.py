import numpy as np


class MultiGaussClassify:
    # self.k: number of class
    # self.d: number of feature
    # self.diag: whether sigma is diagonal
    # self.p[Ci]: prior probability of class i
    # self.C[Ci]: class label for class i
    # self.sigma[Ci][j][l]: covariance matrix of size dxd for class Ci
    # self.inv_sigma[Ci][j][l]: inverse of the covariance matrxi
    # self.u[Ci][j]: mean of class Ci, feature j    
    def __init__(self, k, d, diag=False):  # k: number of class; d: # of features    
        self.k = k
        self.d = d
        self.diag = diag    
        self.p = [1/k]*self.k
        self.u = [[0 for j in range(self.d)] for Ci in range(self.k)]
        self.sigma = [[[0 for k in range(self.d)] for j in range(self.d)] for Ci in range(self.k)]
        self.inv_sigma = [[[0 for k in range(self.d)] for j in range(self.d)] for Ci in range(self.k)]
        
        for Ci in range(0,self.k):
            for j in range(0, self.d):
                for l in range(0, self.d):
                    if j == l:
                        self.sigma[Ci][j][l] = 1
                        self.inv_sigma[Ci][j][l] = 1
                    else:
                        self.sigma[Ci][j][l] = 0
                        self.inv_sigma[Ci][j][l] = 0
        return
        
    def mv_log_gauss(self, Ci, x):
        import math
        z = x - self.u[Ci]
        return (-0.5*np.dot(np.matmul(z,self.inv_sigma[Ci]),z))-np.log(2*math.pi)*(self.d/2.0)-0.5*np.log(np.linalg.det(self.sigma[Ci]))
        
    def fit(self, X, y):
        epsilong = 1.e-6

        self.C = np.unique(y)
        
        if self.C.size != self.k:
            print("missing class labels in the data")
            
        for Ci in range(self.k):
            Xi=[]
            for xi,yi in zip(X,y):
                if yi == self.C[Ci]:
                    Xi.append(xi)
            
            count_in_Xi = len(Xi)
            self.p[Ci] = count_in_Xi/len(X)
            
            Xi = np.transpose(Xi)
            
            for j in range(self.d):
                self.u[Ci][j] = np.average(Xi[j])
                for k in range(count_in_Xi):
                    Xi[j][k] = Xi[j][k] - self.u[Ci][j]          
            
            for j in range(self.d):
                if self.diag:
                    self.sigma[Ci][j][j] = np.dot(Xi[j],Xi[j])/count_in_Xi
                else:
                    for k in range(self.d):
                        self.sigma[Ci][j][k] = np.dot(Xi[j], Xi[k])/count_in_Xi
                        
            self.sigma[Ci] = self.sigma[Ci] + epsilong*np.identity(self.d)
            self.inv_sigma[Ci] = np.linalg.inv(self.sigma[Ci])
        return
        
    def predict(self, X):
        class_label=["null"]*len(X)
        for i in range(len(X)):
            g_Ci_max = -10000000000
            for Ci in range(self.k):
                g_Ci =  self.mv_log_gauss(Ci, X[i]) + np.log(self.p[Ci])
                if g_Ci > g_Ci_max:
                    g_Ci_max = g_Ci
                    class_label[i] = self.C[Ci]
        return class_label
    
    def get_params(self, deep=True):
        return {"k": self.k, "d": self.d, "diag": self.diag}