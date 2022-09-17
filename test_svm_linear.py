import numpy as np
from svm import SVC
from svm import rbf_kernel
import matplotlib.pyplot as plt


# Create dataset
X_train = np.random.randint(-10, 10, size=(700, 2))
y_train = np.sign(np.dot(X_train, [5, 3]))

X_test = np.random.randint(-10, 10, size=(200, 2))
y_test = np.sign(np.dot(X_test, [5, 3]))


# Fit SVM classifier
svc = SVC()
svc.fit(X_train, y_train)


y_pred = svc.predict(X_train)
acc_train = np.mean(np.abs(y_train.flatten() - y_pred.flatten()) < 0.1)
print("Train Accuracy", acc_train)
y_pred = svc.predict(X_test)

x_plus = [x for x, y in zip(X_train, y_train) if y == 1.0]
x_minus = [x for x, y in zip(X_train, y_train) if y == -1.0]

x_plus = np.array(x_plus)
x_minus = np.array(x_minus)


plt.subplot(211)
plt.scatter(x_plus[:, 0], x_plus[:, 1], label="+")
plt.scatter(x_minus[:, 0], x_minus[:, 1], label="-")
plt.legend()
plt.subplot(212)
x_plus = [x for x, y in zip(X_test, y_pred) if y == 1.0]
x_minus = [x for x, y in zip(X_test, y_pred) if y == -1.0]
acc_test = np.mean(np.abs(y_test.flatten() - y_pred.flatten()) < 0.1)
print("Test Accuracy", acc_test)

x_plus = np.array(x_plus)
x_minus = np.array(x_minus)
print(x_plus.shape, x_minus.shape)
plt.scatter(x_plus[:, 0], x_plus[:, 1], label="+")
plt.scatter(x_minus[:, 0], x_minus[:, 1], label="-")
plt.legend()
plt.show()
