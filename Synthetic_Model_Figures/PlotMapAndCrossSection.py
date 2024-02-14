# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 00:26:22 2023

@author: david

Plot the synthetic model map, topography, and cross section.
"""

import geopandas as gp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

map_folder = '../Synthetic_Maps/'
map_file_no_topo = map_folder + 'DoubleFold_NoTopo/DoubleFold_NoTopo.shp'
map_file_topo = map_folder + 'DoubleFold/DoubleFold.shp'
xsec_file = map_folder + 'DoubleFold/DoubleFold_cross_section.shp'
xsec_topo_profile = map_folder + 'DoubleFold/DoubleFold_topo.npz'
axes_file = map_folder+'DoubleFold/DoubleFold_anticline_axes.shp'

colormap = 'viridis_r'
strat_num_shift = 1 #Shift down by this so the youngest unit is 1.
vmin = 1#0
vmax = 7-strat_num_shift
legend_kwds = {'orientation':'vertical',
               'label':'Relative Age',
               'ticks':np.arange(vmin,vmax+1)}

#Plot the map without topography.
map_gdf = gp.GeoDataFrame.from_file(map_file_no_topo, geom_col='geometry', crs='epsg:2154')
map_gdf['RelAge'] -= strat_num_shift
fig1, ax1 = plt.subplots(1, 1)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
map_gdf.plot(ax=ax1, column='RelAge', edgecolor='black',cmap=colormap,cax=cax1,
         categorical=False,legend=True,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
cax1.invert_yaxis()
cax1.text(1.5, 0.93, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax1.transAxes)
cax1.text(1.5, 0.07, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax1.transAxes)
ax1.set_axis_off() #Don't show the axis numbers.
ax1.plot([1.1e4,1.2e4],[1e3,1e3],'k') #Scale bar
ax1.text(1.1e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
#Plot the fold axes.
axes_gdf = gp.GeoDataFrame.from_file(axes_file, geom_col='geometry', crs='epsg:2154')
for i in axes_gdf.index:
    x,y = axes_gdf.geometry[i].xy
    ax1.plot(x,y,'k:')


#Plot the map with topography.
map_gdf2 = gp.GeoDataFrame.from_file(map_file_topo, geom_col='geometry', crs='epsg:2154')
map_gdf2['RelAge'] -= strat_num_shift
fig2, ax2 = plt.subplots(1, 1)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
map_gdf2.plot(ax=ax2, column='RelAge', edgecolor='black',cmap=colormap,cax=cax2,
         categorical=False,legend=True,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
cax2.invert_yaxis()
cax2.text(1.5, 0.93, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax2.transAxes)
cax2.text(1.5, 0.07, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax2.transAxes)
ax2.set_axis_off() #Don't show the axis numbers.
ax2.plot([1.1e4,1.2e4],[1e3,1e3],'k') #Scale bar
ax2.text(1.1e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')

#Add the cross section line to the map with topography plot.
npzfile = np.load(xsec_topo_profile)
[cs_start,cs_end] = npzfile['cs_line']
ax2.plot([cs_start[0],cs_end[0]],[cs_start[1],cs_end[1]],'-k')

#Plot the cross section.
xsec_gdf = gp.GeoDataFrame.from_file(xsec_file, geom_col='geometry', crs='epsg:2154')
xsec_gdf['RelAge'] -= strat_num_shift
fig3, ax3 = plt.subplots(1, 1)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.1)
xsec_gdf.plot(ax=ax3, column='RelAge', edgecolor='black',cmap=colormap,cax=cax3,
         categorical=False,legend=True,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
cax3.invert_yaxis()
cax3.text(1.8, 0.93, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax3.transAxes)
cax3.text(1.8, 0.07, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax3.transAxes)

#Plot the topography.
X = npzfile['X']
Y = npzfile['Y']
topo = npzfile['topo']
fig4, ax4 = plt.subplots(1, 1)
# levels = np.arange(start=3500,stop=4500,step=100)
levels = np.arange(start=3000,stop=3800,step=100)
cs = ax4.contour(X,Y,topo,levels,cmap='gist_earth')
ax4.clabel(cs, inline=True, fontsize=10)
ax4.set_axis_off() #Don't show the axis numbers.
# ax4.set_yticklabels([])
# ax4.set_xticklabels([])
ax4.plot([1.1e4,1.2e4],[1e3,1e3],'k') #Scale bar
ax4.text(1.1e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
ax4.axis('equal') #For the ones plotted with geopandas, this is already true.
plt.plot([0,12760,12760,0,0],[0,0,12760,12760,0],'-k')