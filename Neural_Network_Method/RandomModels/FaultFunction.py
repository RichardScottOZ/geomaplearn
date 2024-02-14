# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:56:03 2023

@author: oakley

This is a function to create an uplift pattern for a vertical, planar, dip-slip fault.
Displacement on the fault surface has an elliptical shape in the strike direction and is constant in the dip direction.
    (Elliptical in both directions might be better, but assuming we aren't dealing with huge elevation differences, this should be okay.)
Displacement decreases quadratically with distance away from the fault as in Cardozo et al. (2008).
Displacement is partitioned between HW and FW according to a parameter alpha.
"""


import numpy as np

def fault(Xin,Yin,cell_size,xc,yc,displacement,length,strike,R,alpha):
    """
    
    Parameters
    ----------
    Xin : numpy array
        x coordinates of points at which uplift should be calculated. (Must be a regular grid in meshgrid format.)
    Yin : numpy array
        y coordinates of points at which uplift should be calculated (Must be a regular grid in meshgrid format.)
    cell_size : float
        Size of the grid cells for Xin and Yin.
    xc: float
        x coordinate of the center of the fault
    yc: float
        y coordinate of the center of the fault
    displacement : float
        Fault displacement. Positive for reverse; negative for normal.
    length : float
        Fault length
    strike : float
        Fault strike direction, in degrees, specified as a Cartesian angle from the x-axis (E), not as an azimuth.
        Note: The HW will be the direction clockwise from the strike (right-hand rule).
    R : float
        The distance perpendicular to the fault that is affected by fault displacement. (Reverse drag radius.)
    alpha : 
        The fraction of displacement in the hanging wall. This should be in the range 0 to 1.
    
    Returns
    -------
    uplift : numpy array
        The amount of uplift at each point in (Xin,Yin)
        
    """
    
    if np.any(np.diff(Xin,axis=0) != cell_size) or np.any(np.diff(Yin,axis=1) != cell_size):
        print('Error: Grid must be in meshgrid format.')
    
    #Rotate to work in a coordinate system with the fault strike direction being the x axis.
    v = strike*np.pi/180 #Convert to radians.
    X = (Xin-xc)*np.cos(-v)-(Yin-yc)*np.sin(-v)
    Y = (Xin-xc)*np.sin(-v)+(Yin-yc)*np.cos(-v)
    
    #Calculate the uplift at each point.
    hw = Y<=0 #Hanging wall.
    r = np.abs(X/length) #Distance in x strike-direction from fault center.
    d = 2*displacement*(1-r)*np.sqrt((1+r)**2/4-r**2) #Displacement on the nearest part of the fault. Wu et al. (2020) Eqn. 6.
    d[r>1] = 0 #Set d to 0 beyond fault tip.
    Y_scaled = np.abs(Y)/R
    D = d*(1-Y_scaled)**2 #Displacement away from the fault surface.
    D [Y_scaled>1] = 0 #Set to 0 beyond the reverse drag radius.
    uplift = hw*alpha*D + (~hw)*(alpha-1)*D #Uplift for the HW and FW (opposite in sign)

    return uplift