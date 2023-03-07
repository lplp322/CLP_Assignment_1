import numpy as np
from math import sqrt
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def train_test_split(x, y, test_size = 0.15):
    test_zero = np.zeros(int(len(x)*test_size))
    dataset = np.concatenate((np.ones(len(x)-len(test_zero)), test_zero))
    np.random.shuffle(dataset)
    feature_train = []
    feature_test =[]
    target_train = []
    target_test = []
    for feature, label, data in zip(x, y, dataset):
        if data == 1:
            feature_train.append(list(feature))
            target_train.append(list(label))
        else:
            feature_test.append(list(feature))
            target_test.append(list(label))
    return np.array(feature_train), np.array(feature_test), np.array(target_train), np.array(target_test)


class K_fold_split:
    def __init__(self, x, y, k=5):
        length = int(len(x) / k)
        folds = np.zeros(length)
        for i in range(1, k - 1):
            fold = np.empty(length)
            fold.fill(i)
            folds = np.concatenate((folds, fold))
        fold_last = np.empty(len(x) - (4 * length))
        fold_last.fill(k - 1)
        folds = np.concatenate((folds, fold_last))
        np.random.shuffle(folds)
        self.x = x
        self.y = y
        self.k = k
        self.val_num = 0
        self.folds = folds

    def get_train_validation(self):
        feature_train = []
        feature_test = []
        target_train = []
        target_test = []
        for feature, label, data in zip(self.x, self.y, self.folds):
            if data != self.val_num:
                feature_train.append(list(feature))
                target_train.append(list(label))
            else:
                feature_test.append(list(feature))
                target_test.append(list(label))
        self.val_num = (self.val_num+1) % self.k
        return np.array(feature_train), np.array(feature_test), np.array(target_train), np.array(target_test)


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
    return accuracy
# def test_numb_of_epochs(n_epochs):
#     features = np.loadtxt("./data/features.txt", delimiter=',')
#     targets = np.loadtxt("./data/targets.txt", dtype='int')
#
#     target_vectors = []
#     for t in targets:
#         v = np.zeros(7)
#         v[t - 1] = 1
#         target_vectors.append(v)
#     target_vectors = np.array(target_vectors)
#
#     feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)
#
#     splitter = K_fold_split(feat_train, target_train)
#     train_f, validation_f, train_l, validation_l = splitter.get_train_validation()
#     train_results = []
#     validation_results = []
#     n_epochs = 10
#     for i in range(n_epochs):
#         train(ann, train_f, train_l)
#         train_results.append(validate(ann, train_f, train_l))
#         validation_results.append(validate(ann, validation_f, validation_l))
#     plt.plot(list(range(n_epochs)), train_results, label='Training')
#     plt.plot(list(range(n_epochs)), validation_results, label='Validation')
#     plt.legend()
#     plt.xlabel('Number of epochs')
#     plt.ylabel('Accuracy')
#     plt.title("Train-validation accuracy")
#     plt.xticks(range(n_epochs))
#     plt.show()
def  accuracies_test():
    features = np.loadtxt("./data/features.txt", delimiter=',')
    targets = np.loadtxt("./data/targets.txt", dtype='int')

    target_vectors = []
    for t in targets:
        v = np.zeros(7)
        v[t - 1] = 1
        target_vectors.append(v)
    target_vectors = np.array(target_vectors)

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)
    accuracy_result = []
    for i in range(10):
        ann = ANN(10, [32], 7, sigmoid, sigmoid_derivative, learning_rate=0.03)
        train(ann, feat_train, target_train)
        accuracy_result.append(validate(ann, feat_test, target_test))
    plt.bar(list(range(10)), accuracy_result)
    plt.xlabel("Tries")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per try")
    plt.show()

if __name__ == '__main__':

    features = np.loadtxt("./data/features.txt", delimiter=',')
    targets = np.loadtxt("./data/targets.txt", dtype='int')

    target_vectors = []
    for t in targets:
        v = np.zeros(7)
        v[t - 1] = 1
        target_vectors.append(v)
    target_vectors = np.array(target_vectors)

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)
    splitter = K_fold_split(feat_train, target_train)
    neuron_performance = []
    for i in range(7, 31, 3):
        ann = ANN(10, [i], 7, sigmoid, sigmoid_derivative, learning_rate=0.01)
        n_epochs = 5

        for i in range(n_epochs):
            train_f, validation_f, train_l, validation_l = splitter.get_train_validation()
            train(ann, feat_train, target_train)

        neuron_performance.append(validate(ann, feat_test, target_test))

    plt.plot(list(range(7, 31, 3)), neuron_performance)
    plt.xlim(7, 31)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.title("Number of neurons vs accuracy")
    plt.show()




