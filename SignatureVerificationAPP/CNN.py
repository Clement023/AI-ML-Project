
import numpy as np
import random
import torch.nn.functional as F



class CNN():

    def __init__(self, si):
        #Here Si is used to denote the number of units in the layers
        self.nl = len(si) #nl is the number of layers
        self.si = si
        #biases will given one vector per layer
        self.biases = [np.random.randn(y,1) for y in si[1:]]
        #zip returns a tuple in which x is the element of the first array
        #and y is the element of the second array
        self.weights = [np.random.randn(y,x) for x,y in zip(si[:-1],si[1:])] 

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def tbatches(self, training_data, batch_size):
        random.shuffle(training_data)
        n = len(training_data)
        # extract data from the training set
        # batches, then, will have several chunks of the main set, each defined by the batch_size_variable
        return [training_data[i:i + batch_size] for i in range(0, n, batch_size)]

    def update(self, batches, alp):
        for i in batches:
            nab_b = [np.zeros(b.shape) for b in self.biases]
            nab_w = [np.zeros(w.shape) for w in self.weights]

            m = len(i)
            # x is a array of n length
            # y is number of elements in x
            for x, y in i:
                del_b, del_w = self.backpropagation(x, y)
                nab_b = [nb + dnb for nb, dnb in zip(nab_b, del_b)]
                nab_w = [nw + dnw for nw, dnw in zip(nab_w, del_w)]

            self.weights = [w - (alp / m) * nw for w, nw in zip(self.weights, nab_w)]
            self.biases = [b - (alp / m) * nb for b, nb in zip(self.biases, nab_b)]

    def backpropagation(self, x, y):
        nab_b = [np.zeros(b.shape) for b in self.biases]
        nab_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            # layer-bound b and w
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward pass
        delt = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nab_b[-1] = delt
        nab_w[-1] = np.dot(delt, activations[-2].transpose())

        for l in range(2, self.nl):
            z = zs[-l]
            sprime = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delt) * sprime
            nabla_b[-l] = delt
            nabla_w[-l] = np.dot(delt, activations[-l-1].transpose())
        return (nab_b, nab_w)

    def sgd(self, training_data, epochs, batch_size, alp, test_data):
        n_test = len(test_data)

        for epoch in range(epochs):
            batches = self.tbatches(training_data, batch_size)
            self.update(batches, alp)

            print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y




def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    


def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))




def distance_metric(features_a, features_b):
    batch_losses = F.pairwise_distance(features_a, features_b)
    return batch_losses






