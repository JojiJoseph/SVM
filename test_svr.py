from svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from svm import rbf_kernel, polynomial_kernel

# Create dataset
X_train = np.random.randint(-10, 10, size=(700, 1))
y_train = np.dot(X_train, [5]) + 5
y_train = (X_train ** 3 ) + 5
y_train = np.sin(X_train*0.5).flatten()
y_train = y_train.flatten()
# print("y_train.shape", y_train.shape)

# X_test = np.random.randint(-10, 10, size=(200, 1))
# y_test = np.dot(X_test, [5]) + 5

svr = SVR(eps=0.1, kernel=polynomial_kernel(d=5))
svr.fit(X_train, y_train)
plt.scatter(X_train, y_train)
x_plot = np.linspace(-10,10,100).reshape((-1,1))
y_pred = svr.predict(x_plot)
# print(y_pred)
plt.plot(x_plot, y_pred)
plt.show()