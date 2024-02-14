# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:37:56 2023

@author: oakley

Make plots of the synthetic maps.
"""

import geopandas as gp
from matplotlib import pyplot as plt

maps_folder = '../Synthetic_Maps/'
names = ['FlatModel','FlatModel_EggBoxTopo','SingleFold','SingleFold_EggBoxTopo',
         'AsymmetricFold','SigmoidalFold','MultipleFolds','MultipleFolds_EggBoxTopo']
titles = ['Flat','Flat - EggBox','Single Flold','Single Fold - EggBox',
          'Asymmetric Fold','Sigmoidal Fold','Multiple Folds','Multiple Folds - EggBox']
crs = 'epsg:2154' #Coordinate reference system
strat_num_lims = [2,6] #Minimum and maximum values of the 'StratNum' column that is used to identify the units.

#Create the figure.
fig, axs = plt.subplots(2, 4)
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, hspace=0.05)
cax = plt.axes([0.1, 0.01, 0.8, 0.07])
legend_kwds = {"label": "Relative Age", "orientation": "horizontal", "ticks": [2,3,4,5,6]}

#Loop through the models.
letters = ['A','B','C','D','E','F','G','H']
for i in range(len(names)):
    ind1 = int(i/4)
    ind2 = int(i%4)
    name = names[i]
    file_path = maps_folder+'/'+name+'/'+name+'.shp'
    geol_map = gp.GeoDataFrame.from_file(file_path, geom_col='geometry', crs=crs)
    if i == 0:
        #We only need to add the legend once.
        legend = True
    else:
        legend = False
    geol_map.plot(ax=axs[ind1,ind2], column='RelAge', edgecolor='black', 
                  legend=legend, cax=cax, legend_kwds=legend_kwds,
                  vmin=strat_num_lims[0],vmax=strat_num_lims[1])
    # print([geol_map['StratNum'].min(),geol_map['StratNum'].max()])
    if ind2 > 0:
        axs[ind1,ind2].yaxis.set_ticklabels([])
    # axs[ind1,ind2].set_title(titles[i])
    axs[ind1,ind2].text(0, 1, letters[i], horizontalalignment='center',
                        verticalalignment='bottom', transform=axs[ind1,ind2].transAxes)
    axs[ind1,ind2].set_axis_off() #Don't show the axis numbers.
    axs[ind1,ind2].plot([8e3,9e3],[-1e3,-1e3],'k') #Scale bar
    axs[ind1,ind2].text(9e3,-1e3,'1 km',horizontalalignment='left',verticalalignment='center',fontsize='small')
cax.invert_xaxis()
cax.text(0.02, -0.9, 'Older', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)
cax.text(0.98, -0.9, 'Younger', horizontalalignment='center',verticalalignment='top',transform=cax.transAxes)
