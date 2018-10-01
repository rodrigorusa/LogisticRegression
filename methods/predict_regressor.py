import numpy as np

from copy import deepcopy

from methods.general import CostCalculus


class PredictRegressor:
    @staticmethod
    def predict(regressors, x, type='onevsall'):
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((ones, x), axis=1)
        classes = np.zeros((x.shape[0], 1))
        for regressor in regressors:
            result = x.dot(regressor['regressor'])
            if type == 'onevsall':
                classes = result.squeeze() == 0
            else:  # multiclassification
                classes = np.argmax(result, axis=0)

        return classes


    def accuracy(y_real, y_pred):
        oks = deepcopy(y_pred)
        idxs = y_real == y_pred
        oks[idxs] = 1
        oks[-idxs] = 0

        return np.count_nonzero(oks)/len(y_real)

