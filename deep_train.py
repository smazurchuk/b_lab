#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:01:22 2017

@author: stephen
"""

# This Notebook is for deep training

import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks
from keras.models import Sequential, Model
from keras.layers import Dense

# Load Data (A little cleaner this time)
data = np.load('data_w_labels.npz')
Bdata = data['vec']     # Binder Word Vectors
Gdata = data['gVec']    # Google word Vectors
L1 = data['L1']     # Super Category labels
L2 = data['L2']     # Category labels

tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

# Lets make a small model first
model1 = Sequential()
model1.add(Dense(65, input_dim=65, kernel_initializer='normal', activation='relu'))
model1.add(Dense(200, input_dim=65, kernel_initializer='normal', activation='relu'))
model1.add(Dense(300, input_dim=200, kernel_initializer='normal'))
model1.compile(loss='cosine_proximity', optimizer='adam')
model1.fit(Bdata, Gdata, epochs=15000, batch_size=535,  verbose=1, callbacks=[tbCallBack])
# Lets see what the error is
M1error = np.sqrt(np.sum((Gdata - model1.predict(Bdata))**2,axis=1) / np.sum(Gdata**2, axis=1))
plt.hist(M1error,15)
plt.title('Simple NN', fontweight='bold', fontsize=16)
plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.savefig('Small_Model.svg', format='svg')

# Lets make a deeper model
model2 = Sequential()
model2.add(Dense(65, input_dim=65, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=65, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(300, input_dim=200, kernel_initializer='normal', activation='relu'))
model2.add(Dense(300, kernel_initializer='normal'))
model2.compile(loss='cosine_proximity', optimizer='adam')
model2.fit(Bdata, Gdata, epochs=15000, batch_size=535,  verbose=1, callbacks=[tbCallBack])
M2error = np.sqrt(np.sum((Gdata - model2.predict(Bdata))**2,axis=1) / np.sum(Gdata**2, axis=1))
plt.hist(M2error,15)
plt.title('Deep NN', fontweight='bold', fontsize=16)
plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.savefig('deppNN.svg', format='svg')

# Lets make a wider model
model3 = Sequential()
model3.add(Dense(65, input_dim=65, kernel_initializer='normal', activation='relu'))
model3.add(Dense(300, input_dim=65, kernel_initializer='normal', activation='relu'))
model3.add(Dense(500, input_dim=300, kernel_initializer='normal', activation='relu'))
model3.add(Dense(300, kernel_initializer='normal'))
model3.compile(loss='cosine_proximity', optimizer='adam')
model2.fit(Bdata, Gdata, epochs=15000, batch_size=535,  verbose=1, callbacks=[tbCallBack])
M3error = np.sqrt(np.sum((Gdata - model3.predict(Bdata))**2,axis=1) / np.sum(Gdata**2, axis=1))
plt.hist(M3error,15)
plt.title('Wide NN', fontweight='bold', fontsize=16)
plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.savefig('wideNN.svg', format='svg')
