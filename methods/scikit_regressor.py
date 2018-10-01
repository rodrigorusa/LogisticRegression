from sklearn import linear_model
from sklearn.metrics import accuracy_score


class ScikitRegressor:
    @staticmethod
    def ovr_regressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):
        return ScikitRegressor.multiclass_regressor('ovr', train_x, train_y, val_x, val_y, max_iterations, learning_rate,
                                                    tolerance)

    @staticmethod
    def multinomial_regressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):
        return ScikitRegressor.multiclass_regressor('multinomial', train_x, train_y, val_x, val_y, max_iterations,
                                                    learning_rate, tolerance)

    @staticmethod
    def multiclass_regressor(multiclass, train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):
        model = linear_model.LogisticRegression(multi_class=multiclass, solver='sag', max_iter=max_iterations
                                                             , tol=tolerance)

        # Fit model
        model.fit(train_x, train_y)

        # Predict
        train_y_pred = model.predict(train_x)
        val_y_pred = model.predict(val_x)

        # Compute error
        train_acc = accuracy_score(train_y, train_y_pred)
        val_acc = accuracy_score(val_y, val_y_pred)

        return model, train_acc, val_acc
