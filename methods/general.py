import numpy as np


class CostCalculus:
    @staticmethod
    def h_theta_logistic(params, variables):
        z = params.dot(variables.T)

        return 1.0/(1.0+np.exp(-1.0*z))

    @staticmethod
    def compute_error_logistic(params, x, y):
        m = x.shape[0]
        h = CostCalculus.h_theta_logistic(params, x)
        tmp = y*np.log(h)+(1.0-y)*np.log(1.0-h)

        return np.sum(tmp)/(-1.0*m)

    @staticmethod
    def h_theta_softmax(params, variables):
        z = params.dot(variables.T)
        z -= np.max(z)
        z_exp = np.exp(z)

        return z_exp/np.sum(z_exp, axis=0)

    @staticmethod
    def compute_error_softmax(params, x, y):
        m = x.shape[0]
        h = CostCalculus.h_theta_softmax(params, x)
        tmp = y * np.log(h) + (1.0 - y) * np.log(1.0 - h)

        return np.sum(tmp)/(-1.0*m)
