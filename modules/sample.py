import numpy as np

class Sample:
    VALUE_KEY = 'VALUE'
    LINEAR_FORECAST_KEY = 'LINEAR_FORECAST'

    def __init__(self, row):
        self.value = row[self.VALUE_KEY]
        self.linear = [row[self.LINEAR_FORECAST_KEY]]

        previous_tags = [ 'Z-{0}'.format(x+1) for x in range(10) ]
        self.previous = np.array(list(map(lambda x : row[x], previous_tags)))

        residuals_tags = [ 'E-{0}'.format(x+1) for x in range(10) ]
        self.residuals = np.array(list(map(lambda x : row[x], residuals_tags)))

        self.encoded = np.array([])

    def get_value(self):
        return self.value

    def set_encoded(self, linear_ws, previous_ws, residual_ws):
        self.encoded = np.array([])

        if linear_ws[0] == 1:
            self.encoded = np.append(self.encoded, self.linear[0])
        
        for i in range(len(previous_ws)):
            if previous_ws[i] == 1:
                self.encoded = np.append(self.encoded, self.previous[i])
        
        for i in range(len(residual_ws)):
            if residual_ws[i] == 1:
                self.encoded = np.append(self.encoded, self.residuals[i])

    def get_encoded(self, linear_ws, previous_ws, residual_ws):
        self.set_encoded(linear_ws, previous_ws, residual_ws)
        return self.encoded