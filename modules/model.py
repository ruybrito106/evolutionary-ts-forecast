from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from svr_params import SVRParams

class Model:
    def __init__(self, X, y, params):
        train_size = 151
        size = len(X)

        self.params = params
        self.train_X, self.train_y = X[:train_size], y[:train_size]
        self.test_X, self.test_y = X[train_size:size], y[train_size:size]

        self.classifier = svm.SVR(epsilon=params.epsilon, kernel=str(params.kernel), C=params.C, gamma=params.gamma, cache_size=1000)

    def train(self):
        self.classifier.fit(self.train_X, self.train_y)

    def test(self):
        self.predictions = list()
        features_size = len(self.test_X[0])

        for t in range(len(self.test_X)):
            input = self.test_X[t].reshape((1, features_size))
            yhat = self.classifier.predict(input)
            self.predictions.append(yhat[0])

        return mean_squared_error(self.test_y, self.predictions)

    def get_result(self):
        return self.test_y, self.predictions

    def grid_search(self):
        # run grid search
        self.classifier = svm.SVR()
        clf = RandomizedSearchCV(self.classifier, param_distributions=SVRParams.grid(), n_iter=30)
        clf.fit(self.train_X, self.train_y)

        # update classifier with best choice
        params = clf.best_params_
        self.classifier = svm.SVR(epsilon=float(params['epsilon']), kernel=str(params['kernel']), C=float(params['C']), gamma=float(params['gamma']), cache_size=1000)

        # return best params
        return {
            'kernel': str(params['kernel']),
            'C': float(params['C']),
            'gamma': float(params['gamma']),
            'epsilon': float(params['epsilon']),
        }