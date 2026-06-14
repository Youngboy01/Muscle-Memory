import numpy as np


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):

        self.lr = lr
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.vw1 = np.zeros_like(self.w1)
        self.vw2 = np.zeros_like(self.w2)
        self.vb1 = np.zeros_like(self.b1)
        self.vb2 = np.zeros_like(self.b2)
        self.momentum = 0.9
        pass

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        probs = self.softmax(self.z2)
        return probs
        pass

    def backward(self, X, y, probs, lambda_l2=0.0001):
        n = y.shape[0]
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), y] = 1
        loss = -np.sum(y_onehot * np.log(probs + 1e-9)) / n
        dz2 = (probs - y_onehot) / n
        dw2 = np.dot(self.a1.T, dz2) + lambda_l2 * self.w2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = np.dot(X.T, dz1) + lambda_l2 * self.w1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        self.vw2 = self.momentum * self.vw2 - self.lr * dw2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2
        self.vw1 = self.momentum * self.vw1 - self.lr * dw1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.w2 += self.vw2
        self.b2 += self.vb2
        self.w1 += self.vw1
        self.b1 += self.vb1
        return loss
        pass

    def train_step(self, X, y):
        probs = self.forward(X)
        loss = self.backward(X, y, probs, lambda_l2=0.0001)
        return loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
