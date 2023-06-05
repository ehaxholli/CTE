import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def import_data(max_size, window_size):

    data=np.loadtxt('001_UCR_Anomaly_35000.txt')[:max_size]
    indexer = np.arange(window_size)[None, :] + np.arange(data.shape[0]+1-window_size)[:, None]
    return data[indexer]

n_plots = len(range(10, 870, 20))
n_cols = min(n_plots, 8)
n_rows = int(np.ceil(2*n_plots / n_cols))

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*5))

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

n_plots = len(1+np.arange(9))
n_cols = min(n_plots, 4)
n_rows = int(np.ceil(2*n_plots / n_cols))

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols*5, n_rows*5))

for i, scale_param in enumerate(1+np.arange(9)):
    print('i: ', i)
    data = import_data(10000, 3)
    percentage_split = 3.3
    int_split = int(100/percentage_split)
    perm = np.random.permutation(data.shape[0])
    data_train = (data[perm])
    data_train = data_train[::int_split]
    gpr = SVR( kernel='poly',gamma='scale', degree=scale_param)
    gpr.fit(data_train[:,:-1], data_train[:,-1])

    row = 2*i // n_cols
    col = (2*i % n_cols) 
    print('row: ', row)
    print('col: ', col)
    
    # Plot train data and predictions
    axs[row, col].plot(data_train[:,-1], alpha=0.5, label='Groung truth')
    axs[row, col].plot(gpr.predict(data_train[:,:-1]), '--', alpha=0.5, label='Prediction')
    axs[row, col].set_title('Degree: {}, data type: train'.format(scale_param))
    axs[row, col].legend()
    
    # Plot test data and predictions
    axs[row, col+1].plot(data[:,-1], alpha=0.5, label='Groung truth')
    axs[row, col+1].plot(gpr.predict(data[:,:-1]), '--', alpha=0.5, label='Prediction')
    axs[row, col+1].set_title('Degree: {}, data type: test'.format(scale_param))
    axs[row, col+1].legend()


    
plt.tight_layout()
plt.savefig('poly_grid.jpg')
plt.show()
plt.clf()
