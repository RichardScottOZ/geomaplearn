# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:21:10 2023

@author: oakley

Plot the different steps in the process of identifying folds with UNET, using the double fold as an example.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

use_topo = True
img_size = (256,256)
img_size_UNET = (256,256)
trained_model = "../Neural_Network_Method/DetectAreasAndAxisModel"
name = 'DoubleFold'

#Load the trained model.
model = tf.keras.models.load_model(trained_model)


#Read in the data.
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
    X[0,0:img_size[0],0:img_size[1],0] = npzfile['unit']
    X[0,0:img_size[0],0:img_size[1],1] = npzfile['topo']
else:
    X[0,0:img_size[0],0:img_size[1],0] = npzfile['unit']
y[0,0:img_size[0],0:img_size[1]] = npzfile['axis'].astype(int) + npzfile['area'].astype(int)
X_orig = X.copy()
                        
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
    
#Predicte the areas and axes for the test dataset.
y_pred = model.predict(X)
#This has size (nmodels,img_size[0],img_size[1],2)
#For the last index, channel 1 is chance of a fold axis, 0 is chance of not a fold axis

#Plot the original data.
plt.figure()
# plt.subplot(1,3,1)
plt.imshow(X_orig[0,:,:,0].T,origin='lower',cmap='viridis_r',vmin=1,vmax=6) #Same vmin and vmax as on the plots of the steps.
plt.axis('equal')
cbar = plt.colorbar()
cbar.ax.invert_yaxis()
plt.title('Relative Age of Units')
plt.xticks([])
plt.yticks([])

plt.figure()
# plt.subplot(1,3,2)
# plt.imshow(X_orig[0,:,:,1].T,origin='lower',cmap='viridis')
plt.imshow(X_orig[0,:,:,1].T,origin='lower',cmap='gist_earth')
plt.axis('equal')
plt.colorbar()
plt.title('Elevation')
plt.xticks([])
plt.yticks([])

plt.figure()
colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, 3)]
cmap_classes = LinearSegmentedColormap.from_list('cmap_classes', colors, N=3)
plt.imshow(y[0,:,:].T,origin='lower',cmap=cmap_classes)
plt.axis('equal')
plt.colorbar(ticks = [0,1,2])
plt.title('True Classification')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(X[0,:,:,0].T,origin='lower',cmap='Greys',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Scaled Relative Ages')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(X[0,:,:,1].T,origin='lower',cmap='Greys',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Scaled Elevation')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(y_pred[0,:,:,0].T,origin='lower',cmap='viridis',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Class 1 Predicted Probability')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(y_pred[0,:,:,1].T,origin='lower',cmap='viridis',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Class 2 Predicted Probability')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(y_pred[0,:,:,2].T,origin='lower',cmap='viridis',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Class 3 Predicted Probability')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(y[0,:,:].T==0,origin='lower',cmap='viridis',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Class 1 Truth')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(y[0,:,:].T==1,origin='lower',cmap='viridis',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Class 2 Truth')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(y[0,:,:].T==2,origin='lower',cmap='viridis',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Class 3 Truth')
plt.xticks([])
plt.yticks([])