import numpy as np

from model import Model
from svr_params import SVRParams

class Individual:
    def __init__(self, samples, linear_weight, prev_weights, prev_residual_weights, params):
        self.samples = samples
        self.fitness = -1.0

        # WEIGHTS
        self.linear_weight = linear_weight
        self.prev_weights = prev_weights
        self.prev_residual_weights = prev_residual_weights
        self.params = params

        if self.hash() > 0:
            # MODEL
            X = list(map(lambda x : x.get_encoded(self.linear_weight, self.prev_weights, self.prev_residual_weights), self.samples))
            y = list(map(lambda x : x.get_value(), self.samples))

            self.model = Model(X, y, params)
            self.model.train()
            self.fitness = self.model.test()

    def hash(self):
        return sum(self.linear_weight) + sum(self.prev_weights) + sum(self.prev_residual_weights)
    
    def set_fitness(self):
        self.fitness = self.model.test()

    def get_fitness(self):
        if self.fitness == -1.0:
            self.set_fitness()
        
        return self.fitness

    def get_results(self):
        return self.model.get_result()

    @staticmethod
    def cross_bit(x, y):
        if np.random.rand() < 0.5:
            return x
        return y

    def cross(self, pair):
        candidate = None
        while True:
            lw = np.array([ self.cross_bit(x, y) for x, y in zip(self.linear_weight, pair.linear_weight) ])
            pw = np.array([ self.cross_bit(x, y) for x, y in zip(self.prev_weights, pair.prev_weights) ])
            prw = np.array([ self.cross_bit(x, y) for x, y in zip(self.prev_residual_weights, pair.prev_residual_weights) ])
            params = self.params.cross(pair.params)

            candidate = Individual(samples=self.samples, linear_weight=lw, prev_weights=pw, prev_residual_weights=prw, params=params)
            if candidate.hash() > 0:
                break

        return candidate

    def mutate_random_toggle(self):
        chosen_feature_random_val = np.random.rand()
        if chosen_feature_random_val < 0.1:
            # 10% toggling linear weight
            self.linear_weight[0] = 1 - self.linear_weight[0]
        elif chosen_feature_random_val < 0.55:
            # 45% toggling some previous weight
            idx = int((chosen_feature_random_val - 0.1) / 0.045)
            self.prev_weights[idx] = 1 - self.prev_weights[idx]
        else:
            # 45% toggling previous error weight
            idx = int((chosen_feature_random_val - 0.55) / 0.045)
            self.prev_residual_weights[idx] = 1 - self.prev_residual_weights[idx]

    def mutate_local_search(self):
        current_fitness = self.get_fitness()

        self.linear_weight[0] = 1 - self.linear_weight[0]
        self.set_fitness()
        if self.fitness <= current_fitness:
            self.linear_weight[0] = 1 - self.linear_weight[0]
            return

        for idx in range(10):
            self.prev_weights[idx] = 1 - self.prev_weights[idx]
            self.set_fitness()
            if self.fitness <= current_fitness:
                self.prev_weights[idx] = 1 - self.prev_weights[idx]
                return
            
            self.prev_residual_weights[idx] = 1 - self.prev_residual_weights[idx]
            self.set_fitness()
            if self.fitness <= current_fitness:
                self.prev_residual_weights[idx] = 1 - self.prev_residual_weights[idx]
                return

    def mutate_grid_search(self):
        best_params = self.model.grid_search()
        self.params = SVRParams(best_params)