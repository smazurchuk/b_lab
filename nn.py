# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 18:40:13 2017

@author: stephen_GAME
"""

import numpy as np
import pandas as pd
import json
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_vec_data():
    # Load Binder Data
    data = pd.read_excel('WordSet1_Ratings.xlsx', 'Sheet1')
    vec = np.zeros([535, 65])
    for row in range(0,535):
        for column in range(5,70):
            if data.iloc[row, column] == "na":
                vec[row-1, column-5] = 0
            else:
                vec[row-1, column-5] = data.iloc[row, column]
    # Load Google data
    with open('gDict.json') as fp:
        gDict = json.load(fp)
    gVec = np.zeros([535,300])
    for idx, word in enumerate(data['Word']):
        gVec[idx,:] = gDict[word][1]
    N, M = vec.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = vec
    all_Y = gVec
    #return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)
    return vec, gVec

X, Y = get_vec_data()
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(65, input_dim=65, init='normal', activation='relu'))
	model.add(Dense(300, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=10, verbose=1)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

model = Sequential()
model.add(Dense(65, input_dim=65, init='normal', activation='relu'))
model.add(Dense(200, input_dim=65, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(200, input_dim=200, init='normal', activation='relu'))
model.add(Dense(300, input_dim=200, init='normal', activation='relu'))
model.add(Dense(300, init='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, nb_epoch=10000, batch_size=535,  verbose=2)

predict1 = model.predict(X[1].reshape(1,65))

p6 = model.predict(X)
err = np.sqrt(np.sum((Y-p6)**2, axis=1)/np.sum(p6**2,axis=1)) 

# Machine learning
from keras.models import Sequential, Model
from keras.layers import Dense

# Lets make a small model first
from keras.models import Sequential, Model
from keras.layers import Dense
model1 = Sequential()
model1.add(Dense(65, input_dim=65, kernel_initializer='normal', activation='relu'))
model1.add(Dense(200, input_dim=65, kernel_initializer='normal', activation='relu'))
model1.add(Dense(300, input_dim=200, kernel_initializer='normal'))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(Bdata, Gdata, epochs=1000, batch_size=535,  verbose=0)

# Can we visualize the model?
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model1).create(prog='dot', format='svg'))

# Lets check the model
import keras
red = np.random.rand(10,20)
blue = np.random.rand(10,20)
lr = keras.losses.mean_squared_error(red,blue)




# Check to make sure
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
