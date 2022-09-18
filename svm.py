import numpy as np
import matplotlib.pyplot as plt
from cvxopt.solvers import qp, coneqp
from cvxopt import matrix


# Uncomment following change settings of the solver
# from cvxopt import solvers
# solvers.options['abstol'] = _____
# solvers.options['reltol'] = _____
# solvers.options['refinement'] = _____
# solvers.options['feastol'] = _____

def rbf_kernel(gamma=1):
    def kernel(x_left, x_right):
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
    def __init__(self, kernel=linear_kernel()) -> None:
        self.kernel = kernel

    def fit(self, X_train, y_train):
        y_train = y_train.reshape((-1, 1))
        P = (y_train @ y_train.T) * self.kernel(X_train, X_train)
        q = -np.ones((len(y_train), 1))
        G = -np.eye(len(y_train))
        h = np.zeros((len(y_train), 1))
        self.X_train = X_train
        self.y_train = y_train
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

    def fit(self, X_train, y_train):
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
        self.X_train = X_train
        self.y_train = y_train
        P = matrix(P.astype(float))
        q = matrix(q.astype(float))
        G = matrix(G.astype(float))
        h = matrix(h.astype(float))
        A = matrix(A.astype(float))
        B = matrix(B)
        alpha = qp(P, q, G, h, A, B)['x']
        alpha = np.array(alpha)

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
        y = np.sum(y, axis=1) +  self.b
        y = np.sign(y).reshape((-1, 1))
        return y

SVC = SVC_soft_margin