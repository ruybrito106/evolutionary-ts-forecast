from pandas import read_csv

from modules.data import Data

class PreProcess:
    def __init__(self, path, lookback_steps, train_total):
        self.series = read_csv(path, header=0, index_col=0)
        self.lookback_steps = int(lookback_steps)
        self.train_total = int(train_total)

    def run(self):
        Data(self.series, self.lookback_steps, self.train_total - self.lookback_steps).process_data()

if __name__ == "__main__":
    import optparse

    class CLI(object):
        def parse_options(self):
            parser = optparse.OptionParser()
            parser.add_option('--p', dest='arg1')
            parser.add_option('--lb', dest='arg2') 
            parser.add_option('--sz', dest='arg3') 
            (self.opt, self.args) = parser.parse_args()

        def run(self):
            self.parse_options()        
            PreProcess(self.opt.arg1, self.opt.arg2, self.opt.arg3).run()
    
    CLI().run()