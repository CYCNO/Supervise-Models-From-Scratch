# Model that return 0 or 1 OR true or false after training from given data

import numpy as np

# create a class with some hyper parameters (learnig_rate, batch_size, no_of_iterations)
class LogisticRegression():
    def __init__(self, learning_rate=0.001, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mean_ = None
        self.std_ = None

    # derivative of this function will be sigmoid(x)* (1 - sigmoid(x))
    # since it's not a linear function we can take it's derivative
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    # using standardize will increase accuracy and minimize loss function by huge margin
    # it makes all the huge value also be treated as others
    def _standardize(self, X):
        return (X - self.mean_) / self.std_
    
    # it use binary cross entropy loss for this one
    # BCE = - 1/N (sum(y_true*log(y_pred) + (1-y_true)*log(1-y_pred)))
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-8 # adding this in log so it don't compute log(0), like log(0) -> error

        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon) 

        return -np.mean(y1 + y2)
    
    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A
    
    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        X = self._standardize(X)

        n_sample, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            A = self.feed_forward(X)
            dz = A - y

            if _ % 100 == 0:
                loss = self.compute_loss(y, A)
                print(f"Iteration {_}, Loss: {loss:.4f}")

            # compute gradient
            dw = (1/n_sample) * np.dot(X.T, dz)
            db = (1/n_sample) * np.sum(A - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = self._standardize(X)
        threshold = .5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)

        return (y_predicted > threshold).astype(int)
    
    def accuracy(self, X_test, Y_test):
        return np.mean(self.predict(X_test) == Y_test)
