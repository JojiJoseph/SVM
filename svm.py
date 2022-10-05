from typing import Union
from unittest.mock import CallableMixin
import numpy as np
import matplotlib.pyplot as plt
from cvxopt.solvers import qp, coneqp
from cvxopt import matrix
from typing import Callable


# Uncomment following change settings of the solver
from cvxopt import solvers
# solvers.options['abstol'] = 1e-100
# solvers.options['reltol'] = 1e-100
# solvers.options['refinement'] = _____
# solvers.options['feastol'] = _____
solvers.options['show_progress'] = False

def rbf_kernel(gamma: float=1):
    def kernel(x_left, x_right) ->np.ndarray:
        res = []
        for x1 in x_left:
            row = []
            for x2 in x_right:
                row.append(np.linalg.norm(x1-x2)**2)
            res.append(row)
        return np.exp(-gamma*np.array(res))
    return kernel


def linear_kernel():
    return lambda x_left, x_right: np.dot(x_left, x_right.T)


def polynomial_kernel(gamma=1, r=1, d=2):
    return lambda x_left, x_right: (gamma*np.dot(x_left, x_right.T)+r)**d


class SVC_hard_margin:
    def __init__(self, kernel: Callable=linear_kernel()) -> None:
        self.kernel = kernel

    def fit(self, X_train: Union[np.ndarray, list], y_train: Union[np.ndarray, list]):
        
        y_train = y_train.reshape((-1, 1))
        P = (y_train @ y_train.T) * self.kernel(X_train, X_train)
        q = -np.ones((len(y_train), 1))
        G = -np.eye(len(y_train))
        h = np.zeros((len(y_train), 1))

        P = matrix(P.astype(float))
        q = matrix(q.astype(float))
        G = matrix(G.astype(float))
        h = matrix(h.astype(float))
        A = y_train.T
        B = 0.
        A = matrix(A.astype(float))
        B = matrix(B)
        alpha = qp(P, q, G, h, A, B)['x']
        # alpha = qp(P, q, G, h)['x']
        alpha = np.array(alpha)
        print(alpha.shape)

        alpha[alpha < 0] = 0
        self.alpha = []
        self.X_train = []
        self.y_train = []
        for x, y, a in zip(X_train, y_train, alpha):
            if a:
                self.X_train.append(x)
                self.y_train.append(y)
                self.alpha.append(a)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train).reshape((-1, 1))
        self.alpha = np.array(self.alpha).reshape((-1, 1))
        self.b = np.mean(self.y_train-np.sum((self.alpha*self.y_train).T *self.kernel(self.X_train, self.X_train), axis=1))

    def predict(self, X):
        y = (self.alpha*self.y_train).T * self.kernel(X, self.X_train)
        y = np.sum(y, axis=1)+ self.b
        y = np.sign(y).reshape((-1, 1))
        return y

        
class SVC_soft_margin:
    def __init__(self, kernel=linear_kernel(), C=0.2) -> None:
        self.kernel = kernel
        self.C = C

    def fit(self, X_train: Union[np.ndarray, list], y_train: Union[np.ndarray, list]) -> None:
        
        # Sanitization of inputs
        if type(X_train) == list:
            X_train = np.array(X_train)
        if type(y_train) == list:
            y_train = np.array(y_train)
        y_train = y_train.reshape((-1, 1))
        
        P = (y_train @ y_train.T) * self.kernel(X_train, X_train)
        q = -np.ones((len(y_train), 1))
        G1 = -np.eye(len(y_train))
        G2 = np.eye(len(y_train))
        h1 = np.zeros((len(y_train), 1))
        h2 = self.C*np.ones((len(y_train), 1))
        G = np.vstack([G1, G2])
        h = np.vstack([h1,h2])
        A = y_train.T
        B = 0.

        P = matrix(P.astype(float))
        q = matrix(q.astype(float))
        G = matrix(G.astype(float))
        h = matrix(h.astype(float))
        A = matrix(A.astype(float))
        B = matrix(B)
        alpha = qp(P, q, G, h, A, B, verbose=False)['x']
        alpha = np.array(alpha)

        alpha[alpha < 0] = 0
        self.alpha_support = []
        self.X_support = []
        self.y_support = []
        for x, y, a in zip(X_train, y_train, alpha):
            if a:
                self.X_support.append(x)
                self.y_support.append(y)
                self.alpha_support.append(a)
        self.X_support = np.array(self.X_support)
        self.y_support = np.array(self.y_support).reshape((-1, 1))
        self.alpha_support = np.array(self.alpha_support).reshape((-1, 1))
        self.b = np.mean(self.y_support-np.sum((self.alpha_support*self.y_support).T *self.kernel(self.X_support, self.X_support), axis=1))

    def predict(self, X: Union[np.ndarray, list])->np.ndarray:
        y = (self.alpha_support*self.y_support).T * self.kernel(X, self.X_support)
        y = np.sum(y, axis=1) +  self.b
        y = np.sign(y).reshape((-1, 1))
        return y

SVC = SVC_soft_margin

class SVC_multiclass:
    def __init__(self, kernel: Callable=linear_kernel(), C: float=1) -> None:
        self.kernel = kernel
        self.C = C
    def fit(self, X_train, y_train):
        data = {}
        for X, y in zip(X_train, y_train):
            y = y[0]
            if y in data:
                data[y].append(X)
            else:
                data[y] = [X]
        n = len(data.keys())
        self.data = data
        self.svms: dict = {}
        for i in range(n-1):
            for j in range(i+1,n):
                svm = SVC(self.kernel, self.C)
                X = np.vstack([data[i],data[j]])
                Y = np.vstack([np.ones((len(data[i]),1)), -np.ones((len(data[j]),1))])
                svm.fit(X,Y)
                self.svms[(i,j)] = svm
    def predict(self, X):
        prediction = []
        n = len(self.data.keys())
        for x in X:
            votes = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if self.svms[(i,j)].predict([x]) > 0:
                        votes[i] += 1
                    else:
                        votes[j] += 1
            prediction.append(np.argmax(votes))
        return prediction 

class SVR:
    def __init__(self, kernel=linear_kernel(), C=10, eps=1) -> None:
        self.kernel = kernel
        self.C = C
        self.eps = eps
    def fit(self, X_train, y_train): # y_train should be flattened
        P = self.kernel(X_train, X_train)
        P = np.hstack([P, -P])
        P = np.vstack([P, -P])
        q = np.hstack([self.eps-y_train, self.eps+y_train]).reshape((-1,1))
        G1 = np.eye(len(q))
        G2 = -G1
        G = np.vstack([G1,G2])

        h = np.vstack([self.C*np.ones((len(y_train)*2,1)),np.zeros((len(y_train)*2,1))])

        A = np.array([1]*len(y_train) + [-1]*len(y_train)).reshape((1,-1))
        B = 0.

        P = matrix(P.astype(float))
        q = matrix(q.astype(float))
        G = matrix(G.astype(float))
        h = matrix(h.astype(float))
        A = matrix(A.astype(float))
        B = matrix(B)
        alpha = qp(P, q, G, h, A, B, verbose=False)['x']
        alpha = np.array(alpha).flatten()
        self.alpha = []
        self.alpha_star = []
        alpha, alpha_star = alpha[:len(alpha)//2], alpha[len(alpha)//2:]
        self.X_support = []
        self.y_support = []
        for a, a_s, X, y in zip(alpha, alpha_star, X_train, y_train):
            if 0 < a <= self.C and 0 < a_s <= self.C:
                self.alpha.append(a)
                self.alpha_star.append(a_s)
                self.X_support.append(X)
                self.y_support.append(y)
        self.alpha = np.array(self.alpha).reshape((-1,1))
        self.alpha_star = np.array(self.alpha_star).reshape((-1,1))
        self.X_support = np.array(self.X_support)
        self.y_support = np.array(self.y_support).reshape((-1,1))
        self.b = np.mean(self.y_support-np.sum((self.alpha-self.alpha_star).T *self.kernel(self.X_support, self.X_support), axis=1))



        
    def predict(self, X):

        y = (self.alpha-self.alpha_star).T * self.kernel(X,self.X_support)

        y = np.sum(y, axis=1) +  self.b


        return y