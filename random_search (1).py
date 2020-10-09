import numpy as np
from numpy import loadtxt
import pandas as pd
import matplotlib.ticker as mtick
import seaborn as sb
import seaborn as sns
import statsmodels.api as sm

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras import backend as K
from keras.layers import Embedding , GlobalAveragePooling1D
from keras.datasets import imdb
from keras.utils.generic_utils import get_custom_objects
keras.backend.set_floatx('float64')

from sklearn import  utils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import *
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn import linear_model
from sklearn.metrics import r2_score
from time import time

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
#%run lr_finder.ipynb


file_name = "BS_data"
data = pd.read_excel("../PRR/" + file_name + ".xlsx")

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def create_model(optimizer = 'adam',activation='relu',init_mode='uniform',dropout_rate=0.1):
# create model
    model = Sequential()
    model.add(Dense(12, input_dim=5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model

model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters



param_grid = {'batch_size':list(range(256, 3000)), 'epochs':list(range(200, 600)),'optimizer':['SGD', 'RMSprop', 'Adam'],
              'init_mode' : ['uniform', 'glorot_uniform', 'he_uniform'],
              'activation': ['ReLu', 'tanh', 'sigmoid', 'elu'],
              'dropout_rate':[0.0, 0.1,0.2]}
grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3, verbose = 5)

grid_result = grid.fit(X, Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



