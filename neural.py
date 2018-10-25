import numpy as np

class NeuralNetwork():
    def __init__(self, inputSize, hiddenSizes, outputSize):
        sizes = [inputSize] + hiddenSizes + [outputSize]

        self.W = []
        for i in range(1, len(sizes)):
            self.W.append(np.random.randn(sizes[i-1], sizes[i]))

        self.b = []
        for i in range(1, len(sizes)):
            self.b.append(np.zeros((1, sizes[i])))

        self.epsilon = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def forward(self, X):
        self.a = [X]
        for i, W in enumerate(self.W):
            z = self.a[i].dot(W) + self.b[i]
            self.a.append(self.sigmoid(z))

        return self.a[-1]

    def backward(self, X, y, output):
        error = y - output
        delta = error*self.sigmoid_derivative(output)
        # print(np.max(abs(delta)))
        self.W[-1] += self.epsilon * self.a[-2].T.dot(delta)
        self.b[-1] += self.epsilon * np.mean(delta)

        for i in reversed(range(0, len(self.W)-1)):
            error = delta.dot(self.W[i+1].T)
            delta = error*self.sigmoid_derivative(self.a[i+1])
            # print(np.max(abs(delta)))
            self.W[i] += self.epsilon * self.a[i].T.dot(delta)
            self.b[i] += self.epsilon * np.mean(delta)

    def train(self, X, y, n=1000):
        for i in range(n):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, XP):
        return self.forward(XP)