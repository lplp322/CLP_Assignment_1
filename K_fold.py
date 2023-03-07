import numpy as np

"""K_fold_split class is used for K_fold cross validation process"""
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

    """splits data into train and validation set, each call gives you next part of dataset as validation set"""
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
        self.val_num = (self.val_num + 1) % self.k
        return np.array(feature_train), np.array(feature_test), np.array(target_train), np.array(target_test)


"""used to split into train and test sets """


def train_test_split(x, y, test_size=0.15):
    test_zero = np.zeros(int(len(x) * test_size))
    dataset = np.concatenate((np.ones(len(x) - len(test_zero)), test_zero))
    np.random.shuffle(dataset)
    feature_train = []
    feature_test = []
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
