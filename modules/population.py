import numpy as np
import random

from keys import Keys
from individual import Individual
from svr_params import SVRParams

class Population:
    def __init__(self, size, samples, retain_ratio=0.2, random_ratio=0.2, mutation_rate=0.05):
        self.size = size
        self.retain_ratio = retain_ratio
        self.random_ratio = random_ratio
        self.mutation_rate = mutation_rate

        self.samples = np.array(samples)
        self.individuals = np.array(self.gen_n_individuals(size, self.samples))
        self.parents = np.array([])

        self.mins = list()

    def hash(self):
        return int(sum(map(lambda x : x.get_fitness(), self.individuals))) % 1024

    def get_results(self):
        return self.mins

    @staticmethod
    def gen_n_individuals(n, samples):
        individuals = list()

        while len(individuals) < n:
            # avoid full empty weights
            candidate = None
            while True:
                lw = np.random.choice([0, 1], 1)

                pw_len = random.randint(1,10)
                pw = np.array([ Population.is_set(x+1, pw_len) for x in range(10) ])

                prw_len = random.randint(1,10)
                prw = np.array([ Population.is_set(x+1, prw_len) for x in range(10) ])

                params = SVRParams()

                candidate = Individual(samples=samples, linear_weight=lw, prev_weights=pw, prev_residual_weights=prw, params=params)
                if candidate.hash() > 0:
                    break

            individuals.append(candidate)

        return individuals

    @staticmethod
    def is_set(a, b):
        return 1 if a <= b else 0

    def top(self):
        fs = list(map(lambda x: x.get_fitness(), self.individuals))
        min_fs = np.min(fs)

        for x in self.individuals:
            if x.get_fitness() == min_fs:
                self.mins.append(x.get_results())
                return x

        return None

    def breed(self):
        """
        Create new generation of individuals from parents
        """

        if len(self.parents) > 1:
            while len(self.individuals) < self.size:
                fst = random.choice(self.parents)
                snd = random.choice(self.parents)
                if fst != snd:
                    self.individuals = np.append(self.individuals, [fst.cross(snd)])

    def select(self):
        """
        Select parents to next generation
        """

        sorted_individuals = list(reversed(sorted(list(self.individuals), key=lambda x: x.get_fitness(), reverse=True)))
        self.individuals = np.array(sorted_individuals)

        # select better fitted
        retain_len = self.retain_ratio * len(self.individuals)
        self.parents = self.individuals[:int(retain_len)]

        # add random new
        random_len = self.random_ratio * len(self.individuals)
        self.parents = np.append(self.parents, self.gen_n_individuals(int(random_len), self.samples))
        self.individuals = np.array([self.individuals[0]])

    def mutate(self):
        """
        Mutate current individuals
        """

        for individual in self.individuals:
            if np.random.rand() <= self.mutation_rate:
                individual.mutate_grid_search()

    def next(self):
        self.select()
        self.breed()
        self.mutate()