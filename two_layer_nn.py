import numpy as np
from cmath import exp


class Perceptron:
    def __init__(self, n_inputs, activation_function, learning_rate):
        self.weights = np.random.normal(0.0, 1.0, n_inputs+1)
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

    def adjust_weight(self, error):
        self.weights[-1] -= self.learning_rate*error
        for i in range(len(self.weights) - 1):
            self.weights[i] -= self.learning_rate*error*self.input_vector[i]



class ANN:

	# # Initialize a network
	# def initialize_network(n_inputs, n_hidden, n_outputs):
	# 	network = list()
	# 	hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	# 	network.append(hidden_layer)
	# 	output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	# 	network.append(output_layer)
	# 	return network

    def __init__(self, n_inputs, hidden_neurons, output_neurons, activation_function, activation_derivative, learning_rate=0.1):
        self.hidden_layer = [Perceptron(n_inputs, activation_function, learning_rate) for i in range(hidden_neurons)]
        self.output_layer = [Perceptron(hidden_neurons, activation_function, learning_rate) for i in range(output_neurons)]
        self.activation_derivative = activation_derivative

    def forward_propagate(self, input_vector):
        hidden_vector = []
        for perceptron in self.hidden_layer:
            hidden_vector.append(perceptron.activate(input_vector))
        result_vector = []
        for perceptron2 in self.output_layer:
            result_vector.append(perceptron2.activate(hidden_vector))
        return np.array(result_vector)

    # label is vector with only one 1 in the place of specified class
    def backward_propagate(self, input_vector, labels):

        estimated_result = self.forward_propagate(input_vector)
        errors_output = []
        for i, perceptron in enumerate(self.output_layer):
            # if labels[i] == 1.0:
            #     error = (perceptron.result-1)*self.activation_derivative(perceptron.result)
            #     perceptron.result * (1 - perceptron.result) * error
            # else:
            #     error = (perceptron.result-0)*self.activation_derivative(perceptron.result)
            error = (perceptron.result - labels[i])*self.activation_derivative(perceptron.result)
            errors_output.append(error)
            perceptron.adjust_weight(error)
        errors_hidden = []
        for i, perceptron in enumerate(self.hidden_layer):
            error = 0.0
            for j, next_perc in enumerate(self.output_layer):
                error = error + next_perc.weights[i] * errors_output[j]
            # error = error/len(self.output_layer)
            errors_hidden.append(error)
            perceptron.adjust_weight(error)


def sigmoid(value):
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


if __name__ == '__main__':
    np.random.seed(0)
    features = np.loadtxt("./data/features.txt", delimiter=',')
    targets = np.loadtxt("./data/targets.txt", dtype='int')
    ann = ANN(10, 10, 7, sigmoid, sigmoid_derivative)

    target_vectors = []
    for t in targets:
        v = np.zeros(7)
        v[t-1] = 1
        target_vectors.append(v)
    target_vectors = np.array(target_vectors)
    from sklearn.model_selection import train_test_split
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