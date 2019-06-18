import numpy as np

class SVRParams:
    def __init__(self, args=None):
        if args == None:
            grid = self.grid()

            self.kernel = np.random.choice(grid['kernel'])
            self.C = np.random.choice(grid['C'])
            self.gamma = np.random.choice(grid['gamma'])
            self.epsilon =  np.random.choice(grid['epsilon'])
        else:
            self.kernel = args['kernel']
            self.C = args['C']
            self.gamma = args['gamma']
            self.epsilon = args['epsilon']

    @staticmethod
    def grid():
        return {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [0.1, 1, 100, 1000],
            'gamma': [1e-1, 1e-2, 1e-4, 1e-7],
            'epsilon': [0, 0.01, 0.3, 1, 4],
        }

    def props(self):
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
        }

    def cross(self, params):
        gamma_ratio = np.random.rand()
        epsilon_ratio = np.random.rand()
        C_ratio = np.random.rand()

        args = {
            'kernel': np.random.choice([params.kernel, self.kernel]),
            'C': (params.C * C_ratio) + (self.C * (1.0 - C_ratio)),
            'gamma': (params.gamma * gamma_ratio) + (self.gamma * (1.0 - gamma_ratio)),
            'epsilon': (params.epsilon * epsilon_ratio) + (self.epsilon * (1.0 - epsilon_ratio)),
        }

        return SVRParams(args)