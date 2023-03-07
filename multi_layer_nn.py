from activation_initialization_functions import *

class Perceptron:

    def __init__(self, n_inputs, activation_function, learning_rate, initial_weights):
        self.weights = initial_weights
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.input_vector = None
        self.result = None
        self.before_activation = None

    def compute_weight(self, input_vector):
        self.input_vector = input_vector
        result = self.weights[-1]  # bias
        for i in range(len(self.weights)-1):
            result += input_vector[i] * self.weights[i]
        self.before_activation = result
        return result

    def activate(self, input_vector):
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
            result_vector.append(perceptron2.compute_weight(input_to_layer))
        result = np.exp(np.array(result_vector)) / np.sum(np.exp(np.array(result_vector)))
        for res, perceptron in zip(result, self.output_layer):
            perceptron.result = res
        return result

    # label is vector with only one 1 in the place of specified class
    def backward_propagate(self, input_vector, labels):
        self.forward_propagate(input_vector)
        errors_output = []
        for i, perceptron in enumerate(self.output_layer):
            error_rate = perceptron.result - labels[i]
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



def train(ann, feat_train, target_train):
    for x, y in zip(feat_train, target_train):
        ann.backward_propagate(x, y)


def validate(ann, feat_test, target_test):
    result = []
    error = 0
    for x, y in zip(feat_test, target_test):
        res = ann.forward_propagate(x)
        result.append(res)
        if res.argmax() != y.argmax():
            error += 1
    accuracy = (len(feat_test)-error)/len(feat_test)
    print("Accuracy: ", accuracy)
    return accuracy, np.array(result)




