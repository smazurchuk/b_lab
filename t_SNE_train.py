#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:09:14 2017

@author: stephen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import callbacks

tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

# Load Data
data = np.load('data/data_w_labels.npz')
Bdata = data['vec']     # Binder Word Vectors
Gdata = data['gVec']    # Google word Vectors
L1 = data['L1']     # Super Category labels
L2 = data['L2']     # Category labels

# Use t-SNE to decompose to 3 dim
B_red = manifold.TSNE(n_components=5).fit_transform(Bdata)
G_red = manifold.TSNE(n_components=5).fit_transform(Gdata)

# Create and Train Models

# Simple NN
model1 = Sequential()
model1.add(Dense(5, input_dim=5))
model1.add(Dense(10, activation='selu'))
model1.add(Dense(5, activation='linear'))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(B_red, G_red, epochs=200000, batch_size=535,  verbose=1, callbacks=[tbCallBack])
M1error = np.sqrt(np.sum((G_red - model1.predict(B_red))**2,axis=1) / np.sum(G_red**2, axis=1))
# plt.hist(M1error,15,edgecolor='k' )
# plt.title('tSNE: Simple NN', fontweight='bold', fontsize=16)
# plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
# plt.ylabel('Frequency',fontsize=16)
# plt.savefig('tSNE_Small_Model.svg', format='svg')

# Deep NN
model2 = Sequential()
model2.add(Dense(5, input_dim=5, kernel_initializer='normal'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='selu'))
model2.add(Dense(5, activation='linear'))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(B_red, G_red, epochs=200000, batch_size=535,  verbose=1, callbacks=[tbCallBack])
M2error = np.sqrt(np.sum((G_red - model2.predict(B_red))**2,axis=1) / np.sum(G_red**2, axis=1))
# plt.hist(M2error,15, edgecolor='k')
# plt.title('tSNE: Deep NN', fontweight='bold', fontsize=16)
# plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
# plt.ylabel('Frequency',fontsize=16)
# plt.savefig('tSNE_deppNN.svg', format='svg')

# Wide NN
model3 = Sequential()
model3.add(Dense(5, input_dim=5, activation='selu'))
model3.add(Dense(30, activation='selu'))
model3.add(Dense(30, activation='selu'))
model3.add(Dense(5, activation='linear'))
model3.compile(loss='mean_squared_error', optimizer='adam')
model3.fit(B_red, G_red, epochs=200000, batch_size=534,  verbose=1, callbacks=[tbCallBack])
M3error = np.sqrt(np.sum((G_red - model3.predict(B_red))**2,axis=1) / np.sum(G_red**2, axis=1))
# plt.hist(M3error,15, edgecolor='k')
# plt.title('Error of NNs', fontweight='bold', fontsize=16)
# plt.xlabel(r'Magnitude of: $ \frac{Error Vector}{Correct Vector}$',fontsize=16)
# plt.ylabel('Frequency',fontsize=16)
# plt.savefig('tSNE_wideNN.svg', format='svg')

np.savez('errors', M1error=M1error, M2error=M2error, M3error=M3error)
