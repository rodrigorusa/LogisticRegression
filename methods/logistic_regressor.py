from copy import deepcopy

import numpy as np

from methods.general import CostCalculus


class LogisticRegressor:
    @staticmethod
    def regressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance,
                          method = 'bgd', type='onevsall'):
        regressors =[]
        classes = np.unique(train_y)

        if (type == 'onevsall'):
            for lbl in classes:
                train_y_log = deepcopy(train_y)
                val_y_log = deepcopy(val_y)
                idxs_train = train_y_log == lbl
                idxs_val = val_y_log == lbl
                train_y_log[idxs_train] = 1
                train_y_log[-idxs_train] = 0
                val_y_log[idxs_val] = 1
                val_y_log[-idxs_val] = 0
                print('Training to class ', lbl)
                if (method == 'bgd'):
                    regressor, _, _, _ = LogisticRegressor.BGDRegressor(train_x, train_y_log, val_x, val_y_log,
                                                                        max_iterations, learning_rate, tolerance,
                                                                        type=type)

                regressors.append({
                    'classification': lbl,
                    'regressor': regressor
                })

        if (type == 'multinomial'):
            train_y_log = np.zeros((len(classes), len(train_y)))
            val_y_log = np.zeros((len(classes), len(val_y)))
            for lbl in classes:
                idx_lbl = int(lbl)
                train_y_log[idx_lbl] = deepcopy(train_y)
                val_y_log[idx_lbl] = deepcopy(val_y)
                idxs_train = train_y_log[idx_lbl] == lbl
                idxs_val = val_y_log[idx_lbl] == lbl
                train_y_log[idx_lbl][idxs_train] = 1
                train_y_log[idx_lbl][~idxs_train] = 0
                val_y_log[idx_lbl][idxs_val] = 1
                val_y_log[idx_lbl][~idxs_val] = 0

            train_y_log = train_y_log.T
            val_y_log = val_y_log.T

            if (method == 'bgd'):
                regressor, _, _, _ = LogisticRegressor.BGDRegressor(train_x, train_y_log, val_x, val_y_log,
                                                                    max_iterations, learning_rate, tolerance,
                                                                    type=type)

            regressors.append({
                'regressor': regressor
            })

        return regressors

    @staticmethod
    def BGDRegressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance, type='onevsall'):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate([ones, train_x], axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        if (type == 'onevsall'):
            # Set random parameters values [0,1) to start
            params = np.random.rand(n)
        if (type == 'multinomial'):
            # Set random parameters values [0,1) to start
            params = np.random.rand(train_y.shape[1], n)

        # Temporary parameters
        tmp_params = np.zeros(n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        k = 1
        while k <= max_iterations:
            if(type == 'onevsall'):
                # Compute model h_theta(x)
                h = CostCalculus.h_theta_logistic(params, train_x)

                # For each variable Xn, calculate the gradient
                tmp = (h - train_y).dot(train_x)
                tmp_params = tmp / m

            if(type == 'multinomial'):
                # Compute model h_theta(x)
                h = CostCalculus.h_theta_softmax(params, train_x)

                # For each variable Xn, calculate the gradient
                h[train_y.T == 1] -= 1
                tmp = h.dot(train_x)
                tmp_params = tmp / m

            # Update coefficients
            params = params - learning_rate * tmp_params

            if (type == 'onevsall'):
                # Compute Error
                train_error[k] = CostCalculus.compute_error_logistic(params, train_x, train_y)

                # Validation Error
                val_error[k] = CostCalculus.compute_error_logistic(params, val_x, val_y)

            if (type == 'multinomial'):
                # Compute Error
                train_error[k] = CostCalculus.compute_error_softmax(params, train_x, train_y)

                # Validation Error
                val_error[k] = CostCalculus.compute_error_softmax(params, val_x, val_y)

            print('Iteration:', k, ', ( Training Error:', train_error[k], ', Validation Error:', val_error[k]), ')'

            # Stop criterion
            if k >= 2:
                if abs(train_error[k-1] - train_error[k]) <= tolerance:
                    break

            k = k + 1

        print(params)

        return params, train_error, val_error, k-1

    @staticmethod
    def SGDRegressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance, type='onevsall'):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        if (type == 'onevsall'):
            # Set random parameters values [0,1) to start
            params = np.random.rand(n)
        if (type == 'multinomial'):
            # Set random parameters values [0,1) to start
            params = np.random.rand(train_y.shape[1], n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        k = 1
        while k <= max_iterations:

            for i in range(0, m):
                h = CostCalculus.h_theta_logistic(params, train_x[i, :])
                # For each variable Xn
                for j in range(0, n):
                    params[j] = params[j] - learning_rate * (h - train_y[i]) * train_x[i, j]

            params_array = np.array([params, ] * m)

            if (type == 'onevsall'):
                # Compute Error
                train_error[k] = CostCalculus.compute_error_logistic(params, train_x, train_y)

                # Validation Error
                val_error[k] = CostCalculus.compute_error_logistic(params, val_x, val_y)

            if (type == 'multinomial'):
                # Compute Error
                train_error[k] = CostCalculus.compute_error_softmax(params, train_x, train_y)

                # Validation Error
                val_error[k] = CostCalculus.compute_error_softmax(params, val_x, val_y)

            print('Iteration:', k, ', ( Training Error:', train_error[k], ', Validation Error:', val_error[k]), ')'

            # Stop criterion
            if k >= 2:
                if abs(train_error[k-1] - train_error[k]) <= tolerance:
                    break

            k = k + 1

        return params, train_error, val_error, k-1

    @staticmethod
    def MBGDRegressor(train_x, train_y, val_x, val_y, b, max_iterations, learning_rate, tolerance, type='onevsall'):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        # Set random parameters values [0,1) to start
        params = np.random.rand(n)
        params = np.array([params, ] * b)

        # Temporary parameters
        tmp_params = np.zeros(n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        k = 1
        while k <= max_iterations:

            for i in range(0, m, b):
                step = b
                if i+b >= m:
                    step = i+b - m
                h = CostCalculus.h_theta_logistic(params[0:step, :], train_x[i:i+step, :], axis=1)
                # For each variable Xn
                for j in range(0, n):
                    tmp = (h - train_y[i:i+step]) * train_x[i:i+step, j]
                    tmp_params[j] = np.sum(tmp) / step

                # Update coefficients
                params[0, :] = params[0, :] - learning_rate * tmp_params
                params = np.array([params[0, :], ] * step)

            params_array = np.array([params[0, :], ] * m)

            if (type == 'onevsall'):
                # Compute Error
                train_error[k] = CostCalculus.compute_error_logistic(params, train_x, train_y)

                # Validation Error
                val_error[k] = CostCalculus.compute_error_logistic(params, val_x, val_y)

            if (type == 'multinomial'):
                # Compute Error
                train_error[k] = CostCalculus.compute_error_softmax(params, train_x, train_y)

                # Validation Error
                val_error[k] = CostCalculus.compute_error_softmax(params, val_x, val_y)

            print('Iteration:', k, ', ( Training Error:', train_error[k], ', Validation Error:', val_error[k]), ')'

            if k >= 2:
                if abs(train_error[k - 1] - train_error[k]) <= tolerance:
                    break

            k = k + 1

        return params[0, :], train_error, val_error, k-1
