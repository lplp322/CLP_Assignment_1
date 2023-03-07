import numpy as np
import matplotlib.pyplot as plt

from K_fold import *
from activation_initialization_functions import *
from multi_layer_nn import *

def prepare_data():
    features = np.loadtxt("./data/features.txt", delimiter=',')
    targets = np.loadtxt("./data/targets.txt", dtype='int')

    target_vectors = []
    for t in targets:
        v = np.zeros(7)
        v[t - 1] = 1
        target_vectors.append(v)
    target_vectors = np.array(target_vectors)
    return features, targets, target_vectors


def numb_of_epochs_test(n_epochs):
    features, targets, target_vectors = prepare_data()

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)

    splitter = K_fold_split(feat_train, target_train)
    train_f, validation_f, train_l, validation_l = splitter.get_train_validation()
    train_results = []
    validation_results = []
    n_epochs = 10
    ann = ANN(10, [32], 7, sigmoid, sigmoid_derivative, learning_rate=0.03)
    for i in range(n_epochs):
        train(ann, train_f, train_l)
        train_results.append(validate(ann, train_f, train_l)[0])
        validation_results.append(validate(ann, validation_f, validation_l)[0])
    plt.plot(list(range(n_epochs)), train_results, label='Training')
    plt.plot(list(range(n_epochs)), validation_results, label='Validation')
    plt.legend()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title("Train-validation accuracy")
    plt.xticks(range(n_epochs))
    plt.show()
def  accuracies_test():
    features, targets, target_vectors = prepare_data()

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)
    accuracy_result = []
    for i in range(10):
        ann = ANN(10, [32], 7, sigmoid, sigmoid_derivative, learning_rate=0.03)
        train(ann, feat_train, target_train)
        accuracy_result.append(validate(ann, feat_test, target_test)[0])
    plt.bar(list(range(10)), accuracy_result)
    plt.ylim(0.85, 0.95)
    plt.xlabel("Tries")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per try")
    plt.show()

def compare_number_of_neurons():
    features, targets, target_vectors = prepare_data()

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)

    neuron_performance = []
    for i in [13, 22, 25, 30]:
        avg_accuracy = 0
        splitter = K_fold_split(feat_train, target_train, k=10)
        for j in range(10):
            train_f, validation_f, train_l, validation_l = splitter.get_train_validation()
            n_epochs = 5
            ann = ANN(10, [i], 7, sigmoid, sigmoid_derivative, learning_rate=0.01)
            for e in range(n_epochs):
                train(ann, train_f, train_l)
            avg_accuracy += validate(ann, validation_f, validation_l)[0]
        avg_accuracy = avg_accuracy / 10
        print("Average accuracy for ", i, "neurons is ", avg_accuracy)
        neuron_performance.append(avg_accuracy)

    plt.plot([14, 22, 25, 30], neuron_performance)
    plt.xlim(7, 31)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.title("Number of neurons vs accuracy")
    plt.show()


def check_the_performance():
    features, targets, target_vectors = prepare_data()

    validation_perf = []
    test_perf = []
    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)
    splitter = K_fold_split(feat_train, target_train, k=5)
    ann = ANN(10, [30], 7, sigmoid, sigmoid_derivative, learning_rate=0.01)
    train_f, validation_f, train_l, validation_l = splitter.get_train_validation()
    for e in range(10):
        train(ann, train_f, train_l)
        validation_perf.append(validate(ann, validation_f, validation_l)[0])
        test_perf.append(validate(ann, feat_test, target_test)[0])
    plt.plot(list(range(10)), validation_perf, label='Validation')
    plt.plot(list(range(10)), test_perf, label='Test')
    plt.legend()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title("Model performance")
    plt.xticks(range(10))
    plt.show()


def create_confusion_matrix():
    features, targets, target_vectors = prepare_data()

    feat_train, feat_test, target_train, target_test = train_test_split(features, target_vectors, test_size=0.15)
    ann = ANN(10, [30], 7, sigmoid, sigmoid_derivative, learning_rate=0.01)
    for i in range(9):
        train(ann, feat_train, target_train)
    res = validate(ann, feat_test, target_test)[1]
    predictions = []
    targets = []
    confusion_matrix = [[0 for i in range(7)] for i in range(7)]
    for i, j in zip(res, target_test):
        predictions.append(i.argmax())
        targets.append(j.argmax())
        confusion_matrix[i.argmax()][j.argmax()] += 1

    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix)
    ax.set_xticks(np.arange(7), labels=list(range(1, 8)))
    ax.set_yticks(np.arange(7), labels=list(range(1, 8)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, "{:.3f}".format(confusion_matrix[i][j]/len(target_test)),
                           ha="center", va="center", color="w")
    plt.xlabel("Target class")
    plt.ylabel("Predicted class")
    plt.title("Confusion matrix")
    plt.show()


if __name__ == '__main__':
    create_confusion_matrix()

