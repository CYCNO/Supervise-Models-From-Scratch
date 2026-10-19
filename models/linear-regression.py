import random

import numpy as np


class LinearRegressionAnalytic:
    """
    Doesn't require training, just put the formula and it will work
    Pros: Very Fast And Accurate
    Cons: Only work for small dataset like <10000
    """

    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.slop = None
        self.intercept = None

    def fit(self):
        if len(self.X) != len(self.Y):
            return "Length Must be same"
        add_X = np.sum(self.X)
        add_Y = np.sum(self.Y)
        add_XX = np.sum(self.X**2)
        add_XY = np.sum(np.multiply(self.X, self.Y))

        A = np.array([[len(self.X), add_X], [add_X, add_XX]])
        B = np.array([add_Y, add_XY])

        fit_val = np.linalg.solve(A, B)

        self.intercept = fit_val[0]
        self.slop = fit_val[1]

        return "Done"

    def predict(self, X):
        if self.slop is None or self.intercept is None:
            return "First Fit The Curve"
        return (self.slop * X) + self.intercept

    def show_line(self, pretty=False):
        if not self.slop or not self.intercept:
            return "First Fit The Curve"

        if pretty:
            sign = "+" if self.intercept > 0 else "-"
            return f"Y = {round(self.slop, 2)}X {sign} {abs(round(self.intercept, 2))}"

        return [self.slop, self.intercept]


class LinearRegressionGD:
    """
    Opposite of `LinearRegressionAnalytics`
    Pros: Can work with very big dataset
    Cons: Slightly less accurate
    """

    def __init__(self, data, iter=1000):
        self.data = data
        self.w = random.uniform(0, 1)  # slope
        self.b = random.uniform(0, 1)  # for b
        self.iter = iter

    def costfn(self, w, b):
        cost = 0
        for i in range(len(self.data)):
            x = self.data[i][0]
            y = self.data[i][1]
            y_pred = w * x + b
            err = y_pred - y
            cost += err**2
        return cost / len(self.data)

    def train(self):
        for i in range(self.iter):
            e = 1e-3
            rate = 1e-3
            dw = (self.costfn(self.w + e, self.b) - self.costfn(self.w, self.b)) / e
            db = (self.costfn(self.w, self.b + e) - self.costfn(self.w, self.b)) / e
            self.w -= rate * dw
            self.b -= rate * db
            print(self.costfn(self.w, self.b))

    def predict(self, x):
        return (self.w * x) + self.b

    def show_param(self):
        return [self.w, self.b]


# ======== Gradient Descent ==========

# data = [[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]]
# lr = LinearRegressionGD(data=data)
# lr.train()
# print(lr.show_param())
# print(lr.predict(5))


# ======== Analytical ==========
# X = [1,2,3,4,6,8]
# Y = [2.4,3,3.6,4,5,6]

# lr = LinearRegressionAnalytic(X, Y)
# lr.fit()
# print(lr.show_line(pretty=True))
