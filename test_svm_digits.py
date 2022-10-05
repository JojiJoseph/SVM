"""Test multiclass SVM"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from svm import SVC_multiclass
import numpy as np
from sklearn.svm import SVC as SK_SVC
from sklearn.metrics import classification_report

X_data, y_data = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

svm = SVC_multiclass()

svm.fit(X_train, y_train)
prediction = svm.predict(X_train)
train_accuracy = np.sum(prediction == y_train.flatten()) / len(y_train)
prediction = svm.predict(X_test)


test_accuracy = np.sum(prediction == y_test.flatten()) / len(y_test)

assert test_accuracy > 0.9

print("\nCustom SVM Results\n----------")
print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)
print(classification_report(y_test, prediction))

print()

svm = SK_SVC(kernel='linear')

svm.fit(X_train, y_train.flatten())
prediction = svm.predict(X_train)
train_accuracy = np.sum(prediction == y_train.flatten()) / len(y_train)
prediction = svm.predict(X_test)

test_accuracy = np.sum(prediction == y_test.flatten()) / len(y_test)

print("\nScikit SVM Results\n----------")
print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

