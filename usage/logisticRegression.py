from utils import *

connect_models_file()

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from models.logisticRegression import LogisticRegression

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23
)

regressor = LogisticRegression(
    learning_rate=0.0001, n_iters=20000
)  # because of small data learning_rate is 0.02
regressor.fit(X_train, y_train)
accuracy = regressor.accuracy(X_test, y_test)
# predict = regressor.predict(X_t)

print("Test accuracy: {0:.3f}".format(accuracy))  # 0.974 = 97% which is good
# print(predict)
