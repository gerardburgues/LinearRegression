
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class LinealRegressionGradDescent:

    def __init__(self, num_params, lr=.1, max_iter=1000, tolerance=1e-3):
        self.w = np.zeros((num_params + 1, 1))
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.cost_list = []

    def predict(self, x):
        return np.dot(x, self.w)

    def inference(self, x):
        x = np.hstack((np.ones((x.shape[0], 1)), x))  # añadimos un 1 a todas las muestras
        return self.predict(x)

    def compute_cost(self, x, y):
        samples = x.shape[0]
        preds = self.predict(x)

        error = preds - y
        cost = 1 / (2 * samples) * np.dot(error.T, error)  # (1/2m)*sum[(error)^2]

        return cost, error

    def update(self, x, error):
        samples = x.shape[0]
        self.w = self.w - (self.lr * (1 / samples) * np.dot(x.T, error))

    def fit(self, x, y):
        x = np.hstack((np.ones((x.shape[0], 1)), x))  # añadimos un 1 a todas las muestras
        old_cost = 1e9
        cost, _ = self.compute_cost(x, y)
        i = 0

        while i < self.max_iter and (old_cost - cost) > self.tolerance:
            i += 1
            cost, error = self.compute_cost(x, y)
            self.update(x, error)

            self.cost_list.append(cost)


# lreg_gd = LinealRegressionGradDescent(10)
# lreg_gd.fit(norm_train_df[['u_q', 'i_d', 'i_q', 'pm', 'ambient', 'torque', 'coolant', 'stator_winding', 'stator_tooth', 'stator_yoke']].values, norm_train_y.reshape(-1, 1))
# print(len(lreg_gd.cost_list), lreg_gd.cost_list[0],lreg_gd.cost_list[-1])
# preds = lreg_gd.inference(norm_test_df[['u_q', 'i_d', 'i_q', 'pm', 'ambient', 'torque', 'coolant', 'stator_winding', 'stator_tooth','stator_yoke']].values)
