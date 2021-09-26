import numpy as np


class LogisticRegression:
    iters = 0
    alpha = 0

    theta_1 = []
    theta_2 = []
    theta_3 = []

    def __init__(self, iters, alpha):
        self.iters = iters
        self.alpha = alpha

    # define sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # predict for test data
    # P (Xt | class = 1)
    def predict(self, Xt, theta):
        z = np.dot(np.transpose(Xt), theta)
        return self.sigmoid(z)

    # define cost function
    # J(theta) = 1/m (-y^T log(h) - (1-y)^T log(1-h) )
    def cost(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)) / y.size

    def gradientDescent(self, X, y):
        theta = np.zeros(X.shape[1])
        np.reshape(theta, (14, 1))

        for i in range(self.iters):
            z = np.dot(X, theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            theta = theta - self.alpha * gradient

        return theta

    def fit(self, X, y):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        y_0 = np.copy(y)
        y_0[y == 2] = 0
        y_0[y == 3] = 0
        self.theta_1 = self.gradientDescent(X, y_0)
        np.reshape(self.theta_1, (14, 1))
        y_1 = np.copy(y)
        y_1[y == 2] = 1
        y_1[y == 1] = 0
        y_1[y == 3] = 0
        self.theta_2 = self.gradientDescent(X, y_1)
        np.reshape(self.theta_2, (14, 1))
        y_2 = np.copy(y)
        y_2[y == 1] = 0
        y_2[y == 2] = 0
        y_2[y == 3] = 1
        self.theta_3 = self.gradientDescent(X, y_2)
        np.reshape(self.theta_3, (14, 1))

    def predictor(self, X_t):
        prediction = []
        preds = [0, 0, 0]
        # print(X_t)
        for j, i in X_t.iterrows():
            # i = X_t.loc[j]
            i = np.concatenate(([1], i))
            np.reshape(i, (14, 1))
            preds[0] = self.predict(i, self.theta_1)
            preds[1] = self.predict(i, self.theta_2)
            preds[2] = self.predict(i, self.theta_3)
            # print(preds)
            ans = max(preds)
            class_label = preds.index(ans) + 1
            # print(class_label)
            prediction.append(class_label)
        return prediction

    def score(self, X_t, y_t):
        prediction = self.predictor(X_t)
        # print(prediction)
        w = 0
        t = 0
        # print(y_t)
        for j in range(len(y_t)):
            # print(type(y_t),prediction[j])
            if y_t.iat[j] != prediction[j]:
                w = w + 1
            t = t + 1
        return (t - w) / t