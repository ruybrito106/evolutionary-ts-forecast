import pandas as pd
import numpy as np

from modules.individual import Individual
from modules.svr_params import SVRParams
from modules.sample import Sample
from modules.keys import Keys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

scaler = StandardScaler()
data = pd.read_csv('data/sunspot_preprocessed.csv', header=0, index_col=0)

scaled = scaler.fit_transform(data.values)
data = pd.DataFrame(scaled, index=data.index, columns=data.columns)

samples = [ Sample(row) for _, row in data.iterrows() ]
linear_ws = np.array([0])
prev_ws = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1])
err_ws = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
params = SVRParams(args={'kernel': 'rbf', 'C': 127.97749523797249, 'gamma': 0.006231931331136936, 'epsilon': 0.14581833528665983})

x = Individual(samples, linear_weight=linear_ws, prev_weights=prev_ws, prev_residual_weights=err_ws, params=params)
vals, preds = x.get_results()

i = data.columns.get_loc(Keys.VALUE_KEY)
s = scaler.scale_[i]
u = scaler.mean_[i]

def revert_scale(arr, s, u):
    return (np.array(arr) * s) + u

val = revert_scale(vals, s, u)
pred = revert_scale(preds, s, u)

mse = mean_squared_error(val, pred)
mae = mean_absolute_error(val, pred)

print (mse, mae)