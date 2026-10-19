import numpy as np

class SVM:
    def __init__(self, lr=0.001, iter=1000, lambda_param=0.01):
        self.lr = lr
        self.iter = iter
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1) # make the y value to -1 or 1
        n_sample, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iter):
            for idx, x_i in enumerate(X):
                output = (x_i @ self.w) - self.b
                if (y_[idx] * output) >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else: # y*output > 1
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

        print("Done")


    def predict(self, X):
        output = X @ self.w - self.b
        return np.sign(output)
