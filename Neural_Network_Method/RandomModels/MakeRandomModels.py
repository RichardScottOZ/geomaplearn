# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:42:15 2023

@author: oakley

Make random fold models.

"""

from FoldFunction import fold
from FaultFunction import fault
import numpy as np
from numpy.random import rand, randn, seed, randint, uniform
import PerlinNoise as pn
import imageio.v3 as iio
import time
# import matplotlib.pyplot as plt

#Set the random number seed for reproducibility.
seed(123)

#Set the number of random models to make.
nmodels = int(1e5+500)

#Define the grid.
cell_size = 50.0 #meters
Nx,Ny = [256,256] #Number of cells in each direction.
xmin,xmax = [0,(Nx-1)*cell_size]
ymin,ymax = [0,(Ny-1)*cell_size]
xvec = np.arange(xmin,xmax+cell_size,cell_size)
yvec = np.arange(ymin,ymax+cell_size,cell_size)
X, Y = np.meshgrid(xvec, yvec, indexing='ij')
z0 = 0.0 #Elevation of top of lowest unit (at the center of the map).
x_center = (xmax+xmin)/2.0
y_center = (ymax+ymin)/2.0
shape = (X.shape)

#Define limits for things that will be chosen randomly.
#Regional parameters:
nhorizon_lims = [5,25]  #Note: This includes infinite top and bottom horizons.
thickness_lims = [50,800]
regional_dip_direction_lims = [0,360]
regional_dip_std = 1.0 #In degrees, assumed to be normally distributed about a mean of 0.
quat_percentile_lims = [0,25] #Percent of the map that is covered by Quaternary deposits.
topo_amplitude_lims = [100,2000]
uplift_noise_amplitude_lims = [0,100] #The amplitude of uplift noise that is added everywhere.
uplift_noise_fraction_lims = [0,0.75] #The maximum fraction of the total uplift that is added / subtracted as noise.

#Fold parameters:
fold_chance = [0.85,0.5,0.3] #Chance of the model having a fold in it for the 1st, 2nd, and 3rd folds.
wavelength_lims = [1000,6000]
plunge_wavelength_lims = [5000,20000]
amplitude_lims = [200,2500]
vergence_lims = [0,360]
xc_lims = [xmin+cell_size,xmax-cell_size]
yc_lims = [ymin+cell_size,ymax-cell_size]
asymmetry_std = 0.3

#Fault parameters:
fault_chance = [0.5,0.3,0.3] #Chance of the model having a fold in it for the 1st, 2nd, and 3rd faults (conditional on the previous one being true).
fault_length_lims = [1500,20000]
DL_scaling_exponent_lims = [-2,0] #Fault displacement will be calculated as fault_length*10**DL_scaling_exponent. -3 to 0 is approximate range from Cowie and Schultz (1992, Fig. 3)
fault_strike_lims = [0,360]
R_fraction_lims = [0.2,0.5] #Reverse drag radius as a fraction of fault length. This is only to one side, so 0.5 means the total fault perpendicular distance affected equals the fault length.
alpha_lims = [0,1] #Controls the distribution of fault slip between the HW and FW. Should never be less than 0 or greater than 1.

#Perlin noise parameters.
octaves_lims = [2,8] #The fewer octaves, the smoother it will be. Note: This will need to be an integer. Also, the runtime increases greatly as this gets larger.
lacunarity_lims = [1,3] #The higher the lacunarity, the more the level of small-scale detail increases with each octave. Should be >1 or detail will decrease.
persistence_lims = [0,1.0] #The higher the persistence, the higher the amplitude of later octaves. Should be <1 or later octaves will be higher amplitude than earlier ones.
frequency_lims = [max(shape)/4,2*max(shape)] #The size of cells in the first octave of the Perlin noise.

def sample_interval(lims):
    #Take a sample from a uniform distribution over a specified interval.
    #lims should be a 2 element list of [min,max]
    sample = uniform(lims[0],lims[1])
    return sample
    
#Keep track of what portion of the dataset falls into each class:
class_counts = np.zeros(3,dtype=np.int64) #Background, fold area, and fold axis.

t1 = time.time()
for n in range(nmodels):
    print(n)
    #Randomly choose the things that apply to the whole map (stratigraphy, regional dip, and topography).
    nhorizons = randint(nhorizon_lims[0],high=nhorizon_lims[1]+1) #The reason for adding +1 to the high value is so that it's inclusive of that value, which otherwise it isn't.
    thicknesses = []
    for i in range(nhorizons-2):
        thicknesses.append(sample_interval(thickness_lims))
    regional_dip_direction = sample_interval(regional_dip_direction_lims)*np.pi/180.0 #Note: This is relative to x axis, not north.
    regional_dip = 0.0+regional_dip_std*randn()*np.pi/180.0
    topo_amplitude = sample_interval(topo_amplitude_lims)
    quat_percentile = sample_interval(quat_percentile_lims)
    uplift_noise_amplitude = uplift_noise_amplitude_lims[0]+rand()*(uplift_noise_amplitude_lims[1]-uplift_noise_amplitude_lims[0])
    uplift_noise_fraction = uplift_noise_fraction_lims[0]+rand()*(uplift_noise_fraction_lims[1]-uplift_noise_fraction_lims[0])
    
    #Randomly add folds.
    axis_in_cell, fold_area = [[],[]]
    uplift, axis_in_cell_sum, fold_area_sum = [np.zeros(X.shape),np.zeros(X.shape,dtype=bool),np.zeros(X.shape,dtype=bool)]
    Nfolds = 0
    for i in range(3):
        has_fold = rand() <= fold_chance[i]
        if has_fold:
            wavelength = sample_interval(wavelength_lims)
            plunge_wavelength = sample_interval(plunge_wavelength_lims)
            amplitude = sample_interval(amplitude_lims)
            vergence = sample_interval(vergence_lims)
            xc = sample_interval(xc_lims)
            yc = sample_interval(yc_lims)
            asymmetry = max(0.0,1.0+asymmetry_std*randn())
            uplift_new, axis_in_cell_new, fold_area_new = fold(X,Y,cell_size,xc,yc,amplitude,wavelength,vergence,
                                                   plunge_wavelength=plunge_wavelength,asymmetry=asymmetry) #Model the fold.
            uplift += uplift_new
            axis_in_cell_sum = axis_in_cell_sum | axis_in_cell_new
            fold_area_sum = fold_area_sum | fold_area_new
            axis_in_cell.append(axis_in_cell_new)
            fold_area.append(fold_area_new)
            Nfolds += 1
        else:
            break
    
    #Randomly add faults.
    #For simplicity, these are purely vertical faults, so they like the fold model used, they cause only uplift without horizontal translation.
    Nfaults = 0
    for i in range(3):
        has_fault = rand() <= fault_chance[i]
        if has_fault:
            xc = sample_interval(xc_lims) #Use the same xc and yc limits as for the folds.
            yc = sample_interval(yc_lims)
            length = sample_interval(fault_length_lims)
            DL_scaling_exponent = sample_interval(DL_scaling_exponent_lims)
            displacement = length*10**DL_scaling_exponent
            strike = sample_interval(fault_strike_lims)
            R_fraction = sample_interval(R_fraction_lims)
            R = length*R_fraction #Reverse drag radius. Do it this way to keep it from being really large relative to length.
            alpha = sample_interval(alpha_lims)
            uplift_new = fault(X,Y,cell_size,xc,yc,displacement,length,strike,R,alpha)
            uplift += uplift_new
            Nfaults += 1
        else:
            break
    
    #Add noise to the uplift.
    #Uplift noise takes the form a+b*u, where u is the true uplift (due to folding and faulting), 
    #a is a normally distributed random value with standard deviation uplift_noise_amplitude, 
    #and b is a normally distributed random value with standard deviation uplift_noise_fraction.
    uplift = uplift+uplift_noise_amplitude*randn(uplift.shape[0],uplift.shape[1])+uplift_noise_fraction*uplift*randn(uplift.shape[0],uplift.shape[1])
    
    #Interpret the regional dip.
    dx = np.cos(regional_dip_direction)*np.cos(regional_dip) #Vector components of the downdip direction.
    dy = np.sin(regional_dip_direction)*np.cos(regional_dip)
    dz = np.sin(regional_dip)
    slope_x = dz/dx
    slope_y = dz/dy
    vert_thicknesses = list(thicknesses/np.cos(regional_dip))
    
    #Randomly choose Perlin noise parameters, including the seed (base).
    #For topography:
    base1 = randint(0,2*nmodels)
    frequency1 = sample_interval(frequency_lims)
    octaves1 = randint(octaves_lims[0],high=octaves_lims[1]+1) #The reason for adding +1 to the high value is so that it's inclusive of that value, which otherwise it isn't.
    lacunarity1 = sample_interval(lacunarity_lims)
    persistence1 = sample_interval(persistence_lims)
    base2 = randint(0,2*nmodels) #Generate a random integer to use as the base.
    frequency2 = sample_interval(frequency_lims)
    octaves2 = randint(octaves_lims[0],high=octaves_lims[1]+1)
    lacunarity2 = sample_interval(lacunarity_lims)
    persistence2 = sample_interval(persistence_lims)
    
    noise = pn.perlin_grid(shape[0], shape[1], frequency=frequency1, base=base1, octaves=octaves1, lacunarity=lacunarity1, persistence=persistence1)
    noise2 = pn.perlin_grid(shape[0], shape[1], frequency=frequency2, base=base2, octaves=octaves2, lacunarity=lacunarity2, persistence=persistence2)
    
    #Convert topography to the appropriate scale
    topo_mean = np.sum(vert_thicknesses)*0.75 #This way it will always be within the stack of layers.
    topo = noise*topo_amplitude+topo_mean
    
    #Calculate at each point on the topography what the original elevation was by subtracting the uplift.
    orig_elev = topo-uplift
    
    #Calculate which unit each topographic point is part of.
    z = np.cumsum([z0]+vert_thicknesses) #orig_elev values for the horizon tops at the center of the map.
    unit = np.zeros(X.shape)
    for i in range(nhorizons):
        if i == 0:
            orig_horizon = slope_x*(X-x_center)+slope_y*(Y-y_center)+z[0]
            mask = orig_elev < orig_horizon
        elif i == nhorizons-1: #-1 b/c the for loop starts with 0.
            orig_horizon = slope_x*(X-x_center)+slope_y*(Y-y_center)+z[-1]
            mask = orig_elev >= orig_horizon
        else:
            orig_horizon1 = slope_x*(X-x_center)+slope_y*(Y-y_center)+z[i-1]
            orig_horizon2 = slope_x*(X-x_center)+slope_y*(Y-y_center)+z[i]
            mask = (orig_elev >= orig_horizon1) & (orig_elev < orig_horizon2)
        unit[mask] = i+1 #Unit numbers start with 1, not 0
    if np.any(unit == 0.0):
        print('Error: Some grid points have no assigned unit.')
    
    #Renumber as relative ages, so lowest number is youngest.
    strat_nums = nhorizons-np.array(unit)+1
    
    #Add some fake quaternary deposits
    quat_cutoff = np.percentile(np.ravel(noise2),quat_percentile)
    strat_nums[noise2<=quat_cutoff] = 0.0
    unit[noise2<=quat_cutoff] = 0.0
    
    #Save the model.
    filename = './Models/Model'+str(n+1)
    iio.imwrite(filename+'_unit.png',strat_nums.astype(int),extension='.png')
    iio.imwrite(filename+'_topo.png',topo.astype(int),extension='.png')
    iio.imwrite(filename+'_axis_sum.png',axis_in_cell_sum.astype(int),extension='.png')
    iio.imwrite(filename+'_area_sum.png',fold_area_sum.astype(int),extension='.png')
    
    class_counts[0] += np.sum((~fold_area_sum) & (~axis_in_cell_sum))
    class_counts[1] += np.sum(fold_area_sum & (~axis_in_cell_sum))
    class_counts[2] += np.sum(axis_in_cell_sum)

    
    # #Make a plot.
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(topo,origin='lower')
    # plt.subplot(2,2,2)
    # plt.imshow(strat_nums,origin='lower')
    # plt.subplot(2,2,3)
    # plt.imshow(axis_in_cell_sum,origin='lower')
    # plt.subplot(2,2,4)
    # plt.imshow(fold_area_sum,origin='lower')
    # plt.title('Model '+str(n+1))
    
t2 = time.time()
print('Created '+str(nmodels)+' models in '+str((t2-t1)/60)+' minutes.')
print('Modeling took '+str((t2-t1)/nmodels)+ ' seconds per model.')
print('Class Percentages:')
class_names = ['Background','Fold Area','Fold Axis']
for i in range(3):
    print('   '+class_names[i]+': '+str(np.round(100*class_counts[i]/np.sum(class_counts),1))+'%')