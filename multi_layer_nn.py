import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split


def sigmoid(value):
    if value < 0:
        return np.exp(value)/(1+np.exp(value))
    else:
        return 1.0 / (1.0 + np.exp(-value))


def sigmoid_derivative(value):
    return value * (1.0 - value)


def relu(value):
    return max(0, value)


def relu_derivative(value):
    if value < 0:
        return 0
    else:
        return 1


def xavier_init(n_input, n_output):
    stddev = np.sqrt(2 / (n_input + n_output))
    return np.random.normal(0, stddev, n_input+1)


def normal_distr(n_values, nothing):
    return np.random.normal(0.0, 1.0, n_values+1)


class Perceptron:

    def __init__(self, n_inputs, activation_function, learning_rate, initial_weights):
        self.weights = initial_weights
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.input_vector = None
        self.result = None
        self.before_activation = None

    def compute_weight(self, input_vector):
        result = self.weights[-1]  # bias
        for i in range(len(self.weights)-1):
            result += input_vector[i] * self.weights[i]
        self.before_activation = result
        return result

    def activate(self, input_vector):
        self.input_vector = input_vector
        value = self.compute_weight(input_vector)
        self.result = self.activation_function(value)
        return self.result

    def adjust_weight(self, error_rate):
        self.weights[-1] -= self.learning_rate * error_rate
        for i in range(len(self.weights) - 1):
            self.weights[i] -= self.learning_rate * error_rate * self.input_vector[i]


class ANN:

    def __init__(self, n_inputs, hidden_neurons, output_neurons, activation_function, activation_derivative,
                 weight_initialisation=normal_distr, learning_rate=0.1):
        self.hidden_layers = []
        input_number = n_inputs
        for layer in hidden_neurons:
            self.hidden_layers.append([Perceptron(input_number, activation_function, learning_rate,
                                                  weight_initialisation(input_number, layer)) for i in range(layer)])
            input_number = layer
        self.output_layer = [Perceptron(input_number, activation_function, learning_rate,
                                        weight_initialisation(input_number, output_neurons)) for i in range(output_neurons)]
        self.activation_derivative = activation_derivative

    def forward_propagate(self, input_vector):
        input_to_layer = input_vector
        for layer in self.hidden_layers:
            hidden_vector = []
            for perceptron in layer:
                hidden_vector.append(perceptron.activate(input_to_layer))
            input_to_layer = hidden_vector
        result_vector = []
        for perceptron2 in self.output_layer:
            result_vector.append(perceptron2.activate(input_to_layer))
        return np.array(result_vector)

    # label is vector with only one 1 in the place of specified class
    def backward_propagate(self, input_vector, labels):
        self.forward_propagate(input_vector)
        errors_output = []
        for i, perceptron in enumerate(self.output_layer):
            error_rate = (perceptron.result - labels[i])*self.activation_derivative(perceptron.result)
            errors_output.append(error_rate)
            perceptron.adjust_weight(error_rate)
        next_layer = self.output_layer
        for layer in reversed(self.hidden_layers):
            errors_hidden = []
            for i, perceptron in enumerate(layer):
                error_rate = 0.0
                for j, next_perc in enumerate(next_layer):
                    error_rate = error_rate + next_perc.weights[i] * errors_output[j]
                errors_hidden.append(error_rate)
                perceptron.adjust_weight(error_rate)
            errors_output = errors_hidden
            next_layer = layer


if __name__ == '__main__':
    np.random.seed(0)
    features = np.loadtxt("./data/features.txt", delimiter=',')
    targets = np.loadtxt("./data/targets.txt", dtype='int')
    ann = ANN(10, [10, 14], 7, sigmoid, sigmoid_derivative, learning_rate = 0.2)

    target_vectors = []
    for t in targets:
        v = np.zeros(7)
        v[t-1] = 1
        target_vectors.append(v)
    target_vectors = np.array(target_vectors)

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.2)


    for x, y in zip(feat_train, target_train):
        ann.backward_propagate(x, y)
    error = 0
    for x, y in zip(feat_test, target_test):
        res = ann.forward_propagate(x)
        if res.argmax() != y.argmax():
            error += 1
    accuracy = (len(target_test)-error)/len(target_test)
    print(accuracy)