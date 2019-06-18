import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

from sample import Sample
from keys import Keys

class Data:
    DST_DATA_PATH = '../data/sunspot_preprocessed.csv'

    def __init__(self, series, lookback_steps, train_size):
        self.series = series
        self.train_size = train_size
        self.lookback_steps = lookback_steps

        size = len(series.values)
        X, y = self.split_sequence(map(lambda x : x[0], series.values.tolist()), lookback_steps)
        
        self.train_X, self.train_y = X[:train_size], y[:train_size]
        self.test_X, self.test_y = X[train_size:size], y[train_size:size]

    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def process_data(self, degree=9, offset=10):
        X = self.series.values
        min_history_length = 60
        a_train, a_test = X[:min_history_length], X[min_history_length:len(X)]
        history = [x for x in a_train]
        predictions = list()

        for i in range(len(a_test)):
            model = ARIMA(history, order=(degree,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast() #pylint: disable=no-member
            yhat = output[0]
            predictions.append([yhat, a_test[i]])
            history.append(a_test[i])

            print ('Done {} / {}'.format(i+1, len(a_test)))

        samples = {
            Keys.LINEAR_FORECAST_KEY: [], 
            Keys.VALUE_KEY: [],
        }

        for i in range(offset):
            samples[Keys.PREVIOUS_VALUE_KEY.format(i+1)] = []
            samples[Keys.PREVIOUS_ERROR_KEY.format(i+1)] = []

        for i in range(offset, len(predictions)):
            lf = predictions[i][0]
            val = predictions[i][1]
            samples[Keys.LINEAR_FORECAST_KEY].append(round(lf[0], 4))
            samples[Keys.VALUE_KEY].append(round(val[0], 4))

            previous = predictions[i-offset:i]
            for j in range(offset):
                prev, err = previous[j][0], previous[j][1]
                samples[Keys.PREVIOUS_VALUE_KEY.format(offset-j)].append(round(prev[0], 4))
                samples[Keys.PREVIOUS_ERROR_KEY.format(offset-j)].append(round(err[0] - prev[0], 4))

        df = pd.DataFrame(samples)
        df.to_csv(self.DST_DATA_PATH, index=True)