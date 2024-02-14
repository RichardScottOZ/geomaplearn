# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:27:37 2023

@author: oakley

This demonstrates how to use a sliding window to interpret a map larger than the 256x256 
image size that the UNET was trained on.

This method was used to interpret the real-world maps shown in the paper. However, 
since those data cannot be redistributed, the method is illustrated here using one 
of the synthetic maps, even though it is not actually needed for this case.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

use_topo = True
img_size = (256,256) #Size that UNET works with.
cell_size = 50 #Pixel size in meters.
trained_model = "./DetectAreasAndAxisModel"
filename = '../Synthetic_Maps/DoubleFold/DoubleFold_raster.npz'
nimg = 1 #Since we are only processing one image at a time.
nclasses = 3

#Load the trained model.
model = tf.keras.models.load_model(trained_model)

#Do some calculations.
half_img_size = (int(img_size[0]/2),int(img_size[1]/2))
quarter_img_size = (int(img_size[0]/4),int(img_size[1]/4))

#Define how much to shift the window by at a time.
#If it takes less than this to reach the end of the map, however, then it will shift by less.
max_shift = min(half_img_size[0],half_img_size[1])

#Load the model.
npzfile = np.load(filename)
units = npzfile['unit'].T
topo = npzfile['topo'].T
#nodata = npzfile['nodata']
nodata = np.zeros(units.shape,dtype=bool) #This may be useful for reading in a real map that has no data pixels, but this synthetic map does not have them.
in_img_size = units.shape

#Rescale the data to the range [0,1].
#For the unit numbers, we first shift the numbers so that the non-Quaternary ones start at 1, while Quaternary is 0.
#For topography, we set the minimum topography to 0 and the maximum to 1.
units_unique = np.unique(units[~nodata]) #These will be sorted from least to greatest.
if not np.any(units_unique==0):
    units_unique = np.concatenate(([0],units_unique))
for i in range(1,len(units_unique)):
    shift = units_unique[i] - units_unique[i-1] - 1
    if shift > 0: #There's a gap.
        units[units >= units_unique[i]] -= shift
        units_unique[i:] -= shift
units = units/units.max()
if use_topo:
    topo = (topo-topo[~nodata].min())/(topo[~nodata].max()-topo[~nodata].min())

#Set both units and topo to 0 in pixels with no data.
units[nodata] = 0
topo[nodata] = 0

#Create the arrays in the form that tensorflow wants.    
if use_topo:
    X = np.zeros((nimg,in_img_size[0],in_img_size[1],2))
    n_channels = 2
    X[0,:,:,0] = units
    X[0,:,:,1] = topo
else:
    X = np.zeros((nimg,in_img_size[0],in_img_size[1],1))
    n_channels = 1
    X[0,:,:,0] = units

#Display the original dimensions.
print('original dimension = ' + str(in_img_size[0]) + 'x' + str(in_img_size[1]))

#Move the sliding window and make predictions in it.
"""
Note: My references to left, right, down, and up here really mean negative 
and positive directions in the first and second dimensions of the image. Depending 
on how the image is made, those may not actually correspond to those directions in the 
way that the image is displayed.
"""
class_probs = np.zeros((nimg,in_img_size[0],in_img_size[1],nclasses))
start_inds = [0,0] #starting indices of the sliding window. These are indices of the image in X.
end_inds = [start_inds[0]+img_size[0],start_inds[1]+img_size[1]]
at_end_x = False
at_end_y = False
pred_made = np.zeros((nimg,in_img_size[0],in_img_size[1]),dtype=bool) #Tells if a prediction has been made yet.
while not at_end_y: #While we haven't reached the top of the image yet.
    while not at_end_x: #While we haven't reached the right side of the image yet.
        X2 = np.zeros((1,img_size[0],img_size[1],X.shape[3]))
        #Calculate how much 0 padding is needed on each side due to the sliding window extending past the edge of the image.
        left_pad = max(-start_inds[0],0)
        right_pad = max(end_inds[0]-in_img_size[0],0)
        bottom_pad = max(-start_inds[1],0)
        top_pad = max(end_inds[1]-in_img_size[1],0)
        X2 = X[:,start_inds[0]:end_inds[0],start_inds[1]:end_inds[1],:]
        #Make the predictions.
        y_pred = model.predict(X2)
        left_ind_mask, right_ind_mask = [quarter_img_size[0],half_img_size[0]+quarter_img_size[0]]
        bottom_ind_mask,top_ind_mask = [quarter_img_size[1],half_img_size[1]+quarter_img_size[1]]
        if start_inds[0] == 0: #Left side
            left_ind_mask = 0
        if end_inds[0] == in_img_size[0]: #Right side
            right_ind_mask = end_inds[0]
        if start_inds[1] == 0: #Bottom
            bottom_ind_mask = 0
        if end_inds[1] == in_img_size[1]: #Top
            top_ind_mask = end_inds[1]
        mask = np.zeros((nimg,img_size[0],img_size[1]),dtype=bool)
        mask[:,left_ind_mask:right_ind_mask,bottom_ind_mask:top_ind_mask] = True
        class_probs[:,start_inds[0]:end_inds[0],start_inds[1]:end_inds[1],:][mask] = y_pred[mask]
        pred_made[:,start_inds[0]:end_inds[0],start_inds[1]:end_inds[1]] = True
        #See if we've reached the end yet.
        at_end_x = end_inds[0] >= in_img_size[0]
        #Shift the window right.
        shift = min(max_shift,in_img_size[0]-end_inds[0])
        start_inds[0] += shift
        end_inds[0] += shift
    #Reset the window to the left side.
    start_inds[0] = 0
    end_inds[0] = start_inds[0]+img_size[0]
    at_end_x = False
    #See if we've reached the end yet.
    at_end_y = end_inds[1] >= in_img_size[1]
    #Shift the window up.
    shift = min(max_shift,in_img_size[1]-end_inds[1])
    start_inds[1] += shift
    end_inds[1] += shift

    
    #Plot the geologic maps and the true and predicted fold axes and areas side by side.
    # plt.subplot(nmaps,3,1+3*n)
    ny = units.shape[0] #Number of pixels
    nx = units.shape[1]
    xbar = 0.9*nx #Since we haven't set extent, this is in pixels, not meters.
    ybar = 0.025*ny
    plt.figure()
    plt.imshow(X[0,:,:,0],origin='lower',cmap='viridis_r')
    plt.plot([xbar,xbar+1e3/cell_size],[ybar,ybar],'k') #Scale bar
    plt.text(xbar,ybar+100/cell_size,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    plt.xticks([]) #For now. We may want something here eventually. Or we may want to add a scale bar.
    plt.yticks([])
    # plt.title('Geology')
    plt.axis('Equal')
    cbar = plt.colorbar()
    cbar.ax.invert_yaxis()
    # plt.subplot(nmaps,3,2+3*n)
    plt.figure()
    plt.imshow(class_probs[0,:,:,2],origin='lower',vmin=0,vmax=1) #Do this to plot area or axis, since axis is really a part of area.
    plt.plot([xbar,xbar+1e3/cell_size],[ybar,ybar],'k') #Scale bar
    plt.text(xbar,ybar+100/cell_size,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    plt.xticks([])
    plt.yticks([])
    # plt.title('Predicted Fold Axis')
    plt.axis('Equal')
    plt.colorbar()
    # plt.subplot(nmaps,3,3+3*n)
    plt.figure()
    plt.imshow(class_probs[0,:,:,1]+class_probs[0,:,:,2],origin='lower',vmin=0,vmax=1) #Do this to plot area or axis, since axis is really a part of area.
    plt.plot([xbar,xbar+1e3/cell_size],[ybar,ybar],'k') #Scale bar
    plt.text(xbar,ybar+100/cell_size,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    plt.xticks([])
    plt.yticks([])
    # plt.title('Predicted Fold Area')
    plt.axis('Equal')
    plt.colorbar()
    plt.figure()
    plt.imshow(X[0,:,:,1],origin='lower',cmap='gist_earth')
    plt.plot([xbar,xbar+1e3/cell_size],[ybar,ybar],'k') #Scale bar
    plt.text(xbar,ybar+100/cell_size,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    plt.xticks([]) #For now. We may want something here eventually. Or we may want to add a scale bar.
    plt.yticks([])
    # plt.title('Elevation')
    plt.axis('Equal')
    plt.colorbar()
