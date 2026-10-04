import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import time


class LogisticRegression(object):

    def __init__(self, k, n, method, alpha=0.001, max_iter=5000,
                 use_penalty=False, l=0.01):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        # Ridge / L2 penalty
        self.use_penalty = use_penalty
        self.l = l
 
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
 
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
 
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])
                batch_X = X[ix:ix + batch_size]
                batch_Y = Y[ix:ix + batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
 
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
 
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
 
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
 
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        # clip away from exact 0/1 to avoid log(0) -> nan in the loss display
        h_clipped = np.clip(h, 1e-15, 1 - 1e-15)
        loss = - np.sum(Y * np.log(h_clipped)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)
 
        # Ridge / L2 penalty 
        # loss term:  + l * sum(theta^2)      
        # grad term:  + l * 2 * theta          
        if self.use_penalty:
            loss = loss + self.l * np.sum(self.W ** 2)
            grad = grad + self.l * 2 * self.W
 
        return loss, grad
 
    def softmax(self, theta_t_x):
        # numerically stable softmax: subtracting the row max doesn't change the result mathematically (it cancels in the ratio), 
        # but keeps every exponent <= 0, preventing overflow as W grows during training
        z = theta_t_x - np.max(theta_t_x, axis=1, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
 
    def softmax_grad(self, X, error):
        return X.T @ error
 
    def h_theta(self, X, W):
        return self.softmax(X @ W)
 
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
 
    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

 
    # Metrics from scratch 
 
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
 
    def precision(self, y_true, y_pred, c):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
 
    def recall(self, y_true, y_pred, c):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
 
    def f1_score(self, y_true, y_pred, c):
        p = self.precision(y_true, y_pred, c)
        r = self.recall(y_true, y_pred, c)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
 
    def macro_precision(self, y_true, y_pred):
        return np.mean([self.precision(y_true, y_pred, c) for c in range(self.k)])
 
    def macro_recall(self, y_true, y_pred):
        return np.mean([self.recall(y_true, y_pred, c) for c in range(self.k)])
 
    def macro_f1(self, y_true, y_pred):
        return np.mean([self.f1_score(y_true, y_pred, c) for c in range(self.k)])
 
    def weighted_precision(self, y_true, y_pred):
        weights = [np.sum(y_true == c) / len(y_true) for c in range(self.k)]
        return sum(w * self.precision(y_true, y_pred, c) for w, c in zip(weights, range(self.k)))
 
    def weighted_recall(self, y_true, y_pred):
        weights = [np.sum(y_true == c) / len(y_true) for c in range(self.k)]
        return sum(w * self.recall(y_true, y_pred, c) for w, c in zip(weights, range(self.k)))
 
    def weighted_f1(self, y_true, y_pred):
        weights = [np.sum(y_true == c) / len(y_true) for c in range(self.k)]
        return sum(w * self.f1_score(y_true, y_pred, c) for w, c in zip(weights, range(self.k)))
 
    def classification_report_scratch(self, y_true, y_pred):
        #Classification_report layout but made from the methods above, for comparison
        print(f"{'':>12}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}")
        for c in range(self.k):
            support = np.sum(y_true == c)
            print(f"{c:>12}{self.precision(y_true,y_pred,c):>12.4f}"
                  f"{self.recall(y_true,y_pred,c):>12.4f}"
                  f"{self.f1_score(y_true,y_pred,c):>12.4f}{support:>12}")
        print()
        print(f"{'accuracy':>12}{'':>24}{self.accuracy(y_true,y_pred):>12.4f}{len(y_true):>12}")
        print(f"{'macro avg':>12}{self.macro_precision(y_true,y_pred):>12.4f}"
              f"{self.macro_recall(y_true,y_pred):>12.4f}"
              f"{self.macro_f1(y_true,y_pred):>12.4f}{len(y_true):>12}")
        print(f"{'weighted avg':>12}{self.weighted_precision(y_true,y_pred):>12.4f}"
              f"{self.weighted_recall(y_true,y_pred):>12.4f}"
              f"{self.weighted_f1(y_true,y_pred):>12.4f}{len(y_true):>12}")
        











