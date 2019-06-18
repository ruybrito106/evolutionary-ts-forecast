import pandas as pd
import numpy as np

import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from modules.population import Population
from modules.sample import Sample
from modules.keys import Keys

class Main:
    def __init__(self, path, population_size, max_epochs, patience):
        self.population_size = int(population_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.scaler = StandardScaler()

        self.data = pd.read_csv(path, header=0, index_col=0)
        scaled = self.scaler.fit_transform(self.data.values)
        self.data = pd.DataFrame(scaled, index=self.data.index, columns=self.data.columns)

        self.samples = []
        for _, row in self.data.iterrows():
            self.samples.append(Sample(row))

    def has_covered(self):
        if len(self.leaders) < self.patience:
            return False
        else:
            sz = len(self.leaders)
            fitness = np.array(map(lambda x : x.get_fitness(), self.leaders[sz-self.patience:sz]))
            return np.amax(fitness) - np.amin(fitness) <= 0.0001    

    def debug_epoch(self, top):
        print ('Epoch {0} ({1}): {2}'.format(self.cur_epoch, self.population.hash(), top.get_fitness()))

    @staticmethod
    def revert_scale(arr, s, u):
        return (np.array(arr) * s) + u

    def compile_results(self):
        print ('Total time: {0} mins'.format((time.time() - self.start_time)/60.0))
        rs = self.population.get_results()

        i = self.data.columns.get_loc(Keys.VALUE_KEY)
        s = self.scaler.scale_[i]
        u = self.scaler.mean_[i]

        errors = list()
        for r in rs:
            value = self.revert_scale(r[0], s, u)
            pred = self.revert_scale(r[1], s, u)
            errors.append(mean_squared_error(value, pred))

        pyplot.plot(errors, color='red')
        pyplot.show()

    def run(self):
        self.start_time = time.time()
        self.population = Population(self.population_size, self.samples)
        self.cur_epoch = 0
        self.leaders = []

        while True:
            top = self.population.top()
            self.debug_epoch(top)
            self.leaders.append(top)
            self.population.next()

            self.cur_epoch = self.cur_epoch + 1
            if self.has_covered() or self.cur_epoch >= self.max_epochs:
                break

        self.compile_results()

if __name__ == "__main__":
    import optparse

    class CLI(object):
        def parse_options(self):
            parser = optparse.OptionParser()
            parser.add_option('--src', dest='arg1')
            parser.add_option('--s', dest='arg2') 
            parser.add_option('--e', dest='arg3') 
            parser.add_option('--p', dest='arg4') 
            (self.opt, self.args) = parser.parse_args()

        def run(self):
            self.parse_options()        
            Main(self.opt.arg1, self.opt.arg2, self.opt.arg3, self.opt.arg4).run()
    
    CLI().run()