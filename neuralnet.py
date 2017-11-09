import numpy as np
from numpy.random import randn
import random


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """Derivative of sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


def cost_prime(x, y):
    """Derivative of cost function"""
    return x - y


class ForwardNeuralNet:

    def __init__(self, sizes):
        """
        'sizes' represents the number of perceptrons within each layer
        the first index of sizes represents the number of features/inputs
        provided and the output of the system will be in the last index
        of the sizes.
        """
        self.sizes = sizes
        self.layer_bias = [randn(size, 1) for size in sizes[1:]]
        self.layer_weights = [randn(y, x)
                              for x, y in zip(sizes[:-1], sizes[1:])]

    def evaluate(self, x):
        """Input a vector x and returns the output of the neural net."""
        x = np.array(x).reshape(-1, 1)
        for bias, weights in zip(self.layer_bias, self.layer_weights):
            x = sigmoid(np.dot(weights, x) + bias)
        return x

    def train(self, training_data, epoch, learning_rate, mini_batch_size):
        """Using simple gradient descent we will attempt to train the
        neural net to minimize the cost function"""
        for i in range(epoch):
            print("%d / %d" % (i, epoch))
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size]
                            for k in range(
                                0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.follow_gradient(mini_batch, learning_rate)

    def follow_gradient(self, training_data, learning_rate):
        bias_change = [np.zeros(b.shape) for b in self.layer_bias]
        weight_change = [np.zeros(w.shape) for w in self.layer_weights]
        for x, y in training_data:
            sbias_change, sweight_change = self.backprop(x, y)
            bias_change = [bc + sbc for bc, sbc in
                           zip(bias_change, sbias_change)]
            # print([i.shape for i in self.layer_weights])
            # print([i.shape for i in sweight_change])
            weight_change = [wc + swc for wc, swc in
                             zip(weight_change, sweight_change)]
        self.layer_weights = [w - np.multiply(
            (learning_rate / len(training_data)), wc)
                              for w, wc in zip(self.layer_weights,
                                               weight_change)]
        self.layer_bias = [b - np.multiply(
            (learning_rate / len(training_data)), bc)
                           for b, bc in zip(self.layer_bias, bias_change)]

    def backprop(self, x, y):
        sbias_change = [np.zeros(b.shape) for b in self.layer_bias]
        sweight_change = [np.zeros(w.shape) for w in self.layer_weights]

        # feedforward
        y = np.array(y).reshape(-1, 1)
        activation = np.array(x).reshape(-1, 1)
        activations = [activation]
        z_list = []  # list to store all the z vectors, layer by layer
        for bias, weight in zip(self.layer_bias, self.layer_weights):
            z = np.dot(weight, activation) + bias
            z_list.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = cost_prime(activations[-1], y) * sigmoid_prime(z_list[-1])
        sbias_change[-1] = delta
        sweight_change[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, len(self.sizes)):
            z = z_list[-l]
            delta = np.dot(self.layer_weights[-l+1].transpose(), delta) \
                * sigmoid_prime(z)
            # print(self.layer_weights[-l + 1].shape, delta.shape, z.shape)
            sbias_change[-l] = delta
            # print(np.array(activations[-l-1]).transpose().shape)
            sweight_change[-l] = np.dot(
                delta, np.array(activations[-l-1]).transpose())
        # print([i.shape for i in sbias_change])
        # print([i.shape for i in sweight_change])
        return sbias_change, sweight_change


if __name__ == '__main__':
    # fnn = ForwardNeuralNet([3, 4, 3, 2, 1])
    # dataset = [([1,1,1], [1]), ([1,1,1], [1]), ([1,1,1], [1]), ([1,1,1], [1]),
    #            ([1,1,1], [1]), ([1,1,1], [1])]
    # print("Bias ", [i.shape for i in fnn.layer_bias])
    # print("Weights ", [i.shape for i in fnn.layer_weights])
    #
    # fnn.train(dataset, 1000, 0.1)
    # print(fnn.evaluate([1,1,1]))
    from sklearn import preprocessing
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fnn = ForwardNeuralNet([4096, 100, 40])
    print("Loading data")
    x = np.loadtxt("train_x.csv", delimiter=",") # load from text
    y = np.loadtxt("train_y.csv", delimiter=",")
    x = x.reshape(-1, 4096) # reshape
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49,
               54, 56, 63, 64, 72, 81]
    lb = preprocessing.LabelBinarizer()
    lb.fit(classes)
    y = lb.transform(y)

    print("Training neural net")
    training_set = list(zip(x, y))
    for i in range(1000):
        print("Epoch %d" % i)
        random.shuffle(training_set)
        fnn.train(training_set[:1000], 1, 0.1, 5000)


    test = np.loadtxt("test_x.csv", delimiter=",")
    test = test.reshape(-1, 4096)
    predictions = []
    for x in test:
        predictions.append(fnn.evaluate(x))
    predictions = np.array(predictions)
    predictions = lb.inverse_transform(predictions)

# with open("test.csv", "w") as file:
#     file.write("Id,Label\n")
#     for i, p in enumerate(predictions, 1):
#         file.write("%d,%d\n"%(i,p[0]))
