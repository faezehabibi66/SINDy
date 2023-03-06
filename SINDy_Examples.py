import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pysindy.feature_library import PolynomialLibrary
import pandas as pd
from pysindy.optimizers import STLSQ
import pysindy as ps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
import scipy.integrate
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------Model---------------------------------------------
def cubic_2D(t, x):
    return [
            -0.1 * x[0] ** 3 + 2 * x[1] ** 3,
            -2 * x[0] ** 3 - 0.1 * x[1] ** 3,
           ]

def lorenz(t, x):
    return [10*(x[1] - x[0]),
            x[0]*(28 - x[2]) - x[1],
            x[0]*x[1] - 8/3*x[2]]


def quadratic(t, x):
 return [-0.1 * x[0] + 2 * x[1] + 0.5* x[0]**2,
         -2 * x[0] - 0.1 * x[1]
         ]


def linear_2D(t, x):
 return [-0.1 * x[0] + 2 * x[1],
         -2 * x[0] - 0.1 * x[1]
         ]


#-----------------------------------------------Data--------------------------------------------
dynModel = linear_2D
x0 = [3,-1.5]

ts = np.arange(0,20,1e-2)

sol = solve_ivp(dynModel, [ts[0], ts[-1]], x0, t_eval=ts)
x = np.transpose(sol.y)
ts = sol.t
#-----------------------------------------------Preprocess--------------------------------------------
#preprocc
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
ts = ts.reshape(-1, 1)
ts = min_max_scaler.fit_transform(ts)
# #Tensor from Numpy
ts = torch.from_numpy(ts)
ts = torch.reshape(ts, (ts.size()[0], 1))
ts = np.squeeze(ts.numpy())

#----------------------------------------------Split: Train-Test--------------------------------------------
x_train = x[:600]
ts_train = ts[:600]

x_test = x[600:]
ts_test = ts[600:]
#----------------------------------------------Model--------------------------------------------
opt = STLSQ(threshold=.01, alpha=.5)
poly_library = PolynomialLibrary(degree=4, include_bias=True)
model = ps.SINDy(feature_library=poly_library, discrete_time=False, optimizer=opt)

model.fit(x_train, t=ts_train)
model.print()

#-----------------------------------------------Data-derivatives--------------------------------------------
dxtrain_dt = np.gradient(x_train, ts_train, axis=0)
dxtest_dt = np.gradient(x_test, ts_test, axis=0)
dxdt_pred_train = model.predict(x_train)
dxdt_pred_test = model.predict(x_test)

#----------------------------------------------Data from model-output--------------------------------------------
def sindy_outmodel(t, x):
    dxdt = model.predict(x.reshape(1, -1))[0]
    return dxdt

# Use solve_ivp() to obtain x from the predicted time derivatives
x_pred = np.zeros_like(x_test)
x_pred[0, :] = x_train[-1].T
for i in range(len(ts_test)-1):
    sol = solve_ivp(sindy_outmodel, [ts_test[i], ts_test[i+1]], x_pred[i, :], t_eval=[ts_test[i+1]])
    x_pred[i+1, :] = sol.y[:, 0]

# Compute the R^2 score of the model's prediction
score = model.score(x_pred, t=ts_test[1]-ts_test[0])
print("Model Prediction Score:", score)
score = model.score(x_test, t=ts_test[1]-ts_test[0])
print("Test Score:", score)
#----------------------------------------------Plot the result--------------------------------------------

plt.plot(ts_test, x_test, label='x_test')
plt.plot(ts_test, x_pred, label='solved SINDy_prediction', linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('dataset')
plt.show()


plt.plot(ts_test, dxtest_dt, label='dx_test/dt')  # Plot the first component of dx/dt
plt.plot(ts_test, dxdt_pred_test, label='SINDy prediction', linestyle='--')  # Plot the first component of dx/dt
plt.legend()
plt.xlabel('Time')
plt.ylabel('Derivative')
plt.show()