# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:34:23 2023

@author: oakley

Plot fold areas and axes detected by the trained UNET vs. the truth.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#I don't understand why, but this is necessary to avoid a crash when plotting.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

use_topo = True
img_size_UNET = (256,256)

Nrepeat = 20#1 #Number of repetitions to average. Only useful if using dropout. Otherwise, set to 1.

#Names of the model files to test on.
names = ['FlatModel','FlatModel_EggBoxTopo','SingleFold','SingleFold_EggBoxTopo',
         'AsymmetricFold','SigmoidalFold','MultipleFolds','MultipleFolds_EggBoxTopo',
         'FoldTrain','DoubleFold']

#Load the trained model.
model = tf.keras.models.load_model("./DetectAreasAndAxisModel")

#Loop through the models.
for i in range(len(names)):
    #Load the model.
    name = names[i]
    if use_topo:
        X = np.zeros((1,img_size_UNET[0],img_size_UNET[1],2))
        n_channels = 2
    else:
        X = np.zeros((1,img_size_UNET[0],img_size_UNET[1],1))
        n_channels = 1
    y = np.zeros((1,img_size_UNET[0],img_size_UNET[1]))
    filename = '../Synthetic_Maps/'+name+'/'+name+'_raster.npz'
    npzfile = np.load(filename)
    if use_topo:
        X[0,:,:,0] = npzfile['unit']
        X[0,:,:,1] = npzfile['topo']
    else:
        X[0,:,:,0] = npzfile['unit']
    y[0,:,:] = npzfile['axis'].astype(int) + npzfile['area'].astype(int)
                        
    #Rescale the data to the range [0,1].
    #For the unit numbers, we first shift the numbers so that the non-Quaternary ones start at 1, while Quaternary is 0.
    #For topography, we set the minimum topography to 0 and the maximum to 1.
    units_unique = np.unique(X[0,:,:,0])
    shift = np.min(units_unique[units_unique != 0])-1
    X[0,:,:,0][X[0,:,:,0] != 0.0] -= shift
    for j in range(1,len(units_unique)):
        shift = units_unique[j] - units_unique[j-1] - 1
        if shift > 0: #There's a gap.
            X[0,:,:,0][X[0,:,:,0] >= units_unique[j]] -= shift
            units_unique[j:] -= shift
    X[0,:,:,0] = X[0,:,:,0]/X[0,:,:,0].max()
    if use_topo:
        X[0,:,:,1] = (X[0,:,:,1]-X[0,:,:,1].min())/(X[0,:,:,1].max()-X[0,:,:,1].min())
        # X[0,:,:,1] = (X[0,:,:,1]-X[0,:,:,1].min())/1e3 #Try consistent scaling of the topography.
        # X[0,:,:,1] = (X[0,:,:,1]-X[0,:,:,1].mean())/600 #Try consistent scaling of the topography around its mean.
    
    # Predicte the axes and fold areas for the test dataset.
    y_pred = model.predict(X)
    
    #Plot the true and predicted fold axes and areas side by side.
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(y[0,:,:].T,origin='lower')
    plt.title('True Fold Area and Axis')
    plt.subplot(2,2,2)
    plt.imshow(y_pred[0,:,:,1].T+y_pred[0,:,:,2].T,origin='lower') #Do this to plot area or axis, since axis is really a part of area.
    plt.clim(0,1)
    plt.colorbar()
    plt.title('Predicted Fold Area')
    plt.subplot(2,2,3)
    plt.imshow(y_pred[0,:,:,2].T,origin='lower')
    plt.colorbar()
    plt.title('Predicted Fold Axis')
    plt.subplot(2,2,4)
    plt.title(name)
    plt.imshow(X[0,:,:,0].T,origin='lower')