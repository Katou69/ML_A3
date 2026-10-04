import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class LinearRegression(object):

    kfold = KFold(n_splits=5)

    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=500,
                 bs=50, cv=kfold, init='zeros', momentum=0, use_momentum=False):
        self.lr = lr
        self.num_epochs = num_epochs
        self.bs = bs
        self.method = method
        self.cv = cv
        self.regularization = regularization
        self.init = init
        self.momentum = momentum
        self.use_momentum = use_momentum

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def fit(self, X_train, y_train):
        self.kfold_scores = list()
        self.val_loss_old = np.inf

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val = X_train[val_idx]
            y_cross_val = y_train[val_idx]

            m = X_cross_train.shape[1]
            if self.init == 'xavier':
                lower, upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m))
                self.theta = np.random.uniform(low=lower, high=upper, size=m)
            else:
                self.theta = np.zeros(m)

            self.prev_step = np.zeros(m)

            for epoch in range(self.num_epochs):
                perm = np.random.permutation(X_cross_train.shape[0])
                X_cross_train = X_cross_train[perm]
                y_cross_train = y_cross_train[perm]

                if self.method == 'mini':
                    for batch_idx in range(0, X_cross_train.shape[0], self.bs):
                        X_method_train = X_cross_train[batch_idx:batch_idx + self.bs, :]
                        y_method_train = y_cross_train[batch_idx:batch_idx + self.bs]
                        train_loss = self._train(X_method_train, y_method_train)
                elif self.method == 'sto':
                    for i in range(X_cross_train.shape[0]):
                        X_method_train = X_cross_train[i:i + 1, :]
                        y_method_train = y_cross_train[i:i + 1]
                        train_loss = self._train(X_method_train, y_method_train)
                else:
                    X_method_train = X_cross_train
                    y_method_train = y_cross_train
                    train_loss = self._train(X_method_train, y_method_train)

                yhat_val = self.predict(X_cross_val)
                val_loss_new = self.mse(y_cross_val, yhat_val)

                if np.allclose(val_loss_new, self.val_loss_old):
                    break
                self.val_loss_old = val_loss_new

            self.kfold_scores.append(val_loss_new)
            print(f"Fold {fold}: {val_loss_new}")

    def _train(self, X, y):
        yhat = self.predict(X)
        m = X.shape[0]
        grad = (1 / m) * X.T @ (yhat - y) + self.regularization.derivation(self.theta)

        step = self.lr * grad
        if self.use_momentum:
            self.theta = self.theta - step + self.momentum * self.prev_step
            self.prev_step = step
        else:
            self.theta = self.theta - step

        return self.mse(y, yhat)

    def predict(self, X):
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]

    def r2(self, ytrue, ypred):
        ss_res = sum((ytrue - ypred) ** 2)
        ss_tot = sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_feature_importance(self, feature_names):
        coefficients = self._coef()
        importance_order = np.argsort(np.abs(coefficients))[::-1]
        sorted_names = np.array(feature_names)[importance_order]
        sorted_coefficients = coefficients[importance_order]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_names, sorted_coefficients)
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class Lasso:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class Ridge:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class Normal:
    def __init__(self, l=0):
        self.l = l

    def __call__(self, theta):
        return 0

    def derivation(self, theta):
        return 0


class RidgeRegression(LinearRegression):
    def __init__(self, method, lr, l, init='zeros', momentum=0, use_momentum=False, num_epochs=500):
        self.regularization = Ridge(l)
        super().__init__(self.regularization, lr, method, num_epochs=num_epochs,
                          init=init, momentum=momentum, use_momentum=use_momentum)


class LassoRegression(LinearRegression):
    def __init__(self, method, lr, l, init='zeros', momentum=0, use_momentum=False, num_epochs=500):
        self.regularization = Lasso(l)
        super().__init__(self.regularization, lr, method, num_epochs=num_epochs,
                          init=init, momentum=momentum, use_momentum=use_momentum)


class NormalRegression(LinearRegression):
    def __init__(self, method, lr, init='zeros', momentum=0, use_momentum=False, num_epochs=500):
        self.regularization = Normal()
        super().__init__(self.regularization, lr, method, num_epochs=num_epochs,
                          init=init, momentum=momentum, use_momentum=use_momentum)
        











