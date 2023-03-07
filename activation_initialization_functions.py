import numpy as np

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
