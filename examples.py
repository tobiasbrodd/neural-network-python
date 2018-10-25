import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV
from neural import NeuralNetwork

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.round(Z.reshape(xx.shape))
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=plt.cm.Spectral)

def plot_predictions(predict, X, y, X_scale, y_scale):
    X *= X_scale
    y *= y_scale
    plt.plot(X, y, color="red")
    XP = np.arange(0, X_scale).reshape((X_scale, 1)) / X_scale
    yp = predict(XP)
    XP *= X_scale
    yp *= y_scale
    plt.scatter(XP, yp, color="limegreen", marker='.')

def fixed_sequence_demo():
    inputSize = 1
    hiddenSizes = [100, 200, 100]
    outputSize = 1

    sequence = np.arange(0, 10, 0.1)

    X = np.array([sequence]).T
    y = np.array([np.exp(sequence)]).T

    X_scale = np.power(10, len(str(int(np.amax(X)))))
    y_scale = np.power(10, len(str(int(np.amax(y)))))

    X = X / X_scale
    y = y / y_scale

    NN = NeuralNetwork(inputSize, hiddenSizes, outputSize)
    NN.train(X, y, 10000)

    plot_predictions(lambda x: NN.predict(x), X, y, X_scale, y_scale)
    plt.show()

def normal_sequence_demo():
    inputSize = 1
    hiddenSizes = [25]
    outputSize = 1

    for i in range(0, 4):
        X = np.array([range(0, 100)]).T
        y = [1]
        values = [1]
        for j in range(1, 100):
            y.append(y[-1] + np.random.normal(0, 1))
        y = np.array([y]).T

        X_scale = np.power(10, len(str(int(np.amax(X)))))
        y_scale = np.power(10, len(str(int(np.amax(y)))))

        X = X / X_scale
        y = y / y_scale

        NN = NeuralNetwork(inputSize, hiddenSizes, outputSize)
        NN.train(X, y, 10000)

        plt.subplot(221 + i)
        plot_predictions(lambda x: NN.predict(x), X, y, X_scale, y_scale)
    plt.show()

def random_sequence_demo():
    inputSize = 1
    hiddenSizes = [25]
    outputSize = 1

    for i in range(0, 4):
        X = np.sort(np.random.uniform(0, 100, (25, inputSize)), axis=0)
        X = np.sort(np.concatenate((X, X * 0.75)), axis=0)
        y = np.sort(np.random.uniform(0, 100, (25, outputSize)), axis=1)
        y = np.sort(np.concatenate((y, y * 0.75)), axis=1)

        X_scale = np.power(10, len(str(int(np.amax(X)))))
        y_scale = np.power(10, len(str(int(np.amax(y)))))

        X = X / X_scale
        y = y / y_scale

        NN = NeuralNetwork(inputSize, hiddenSizes, outputSize)
        NN.train(X, y, 10000)

        plt.subplot(221 + i)
        plot_predictions(lambda x: NN.predict(x), X, y, X_scale, y_scale)
    plt.show()

def logistic_regression_decision_boundary_demo():
    np.random.seed(0)
    X, y = make_moons(200, noise=0.20)
    clf = LogisticRegressionCV()
    clf.fit(X, y)
    y = np.array([y]).T
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.show()

def moons_decision_boundary_demo():
    inputSize = 2
    hiddenSizes = [5, 10, 5]
    outputSize = 1

    np.random.seed(0)
    X, y = make_moons(200, noise=0.20)
    y = np.array([y]).T
    NN = NeuralNetwork(inputSize, hiddenSizes, outputSize)
    NN.train(X, y, 10000)
    plot_decision_boundary(lambda x: NN.predict(x), X, y)
    plt.show()

def random_decision_boundary_demo():
    inputSize = 2
    hiddenSizes = [10, 20, 30, 20, 10]
    outputSize = 1

    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, (100, 1))
    NN = NeuralNetwork(inputSize, hiddenSizes, outputSize)
    NN.train(X, y, 10000)
    plot_decision_boundary(lambda x: NN.predict(x), X, y)
    plt.show()

if __name__ == '__main__':
    # fixed_sequence_demo()
    # normal_sequence_demo()
    # random_sequence_demo()
    # logistic_regression_decision_boundary_demo()
    moons_decision_boundary_demo()
    # random_decision_boundary_demo()
    