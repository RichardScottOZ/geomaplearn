# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:50:05 2023

@author: oakley

Make a series of plots showing the process of clustering-based fold identification for a single map.

Useful references for plotting with geopandas:
    https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.plot.html
    https://geopandas.org/en/stable/docs/user_guide/mapping.html
    
"""

import geopandas as gp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np

# map_name = 'SingleFold'
map_name = 'DoubleFold'
map_folder = '../Synthetic_Maps/'
results_folder = '../Unsupervised_Clustering_Method/tmp/'
nodes = 20 #Number of nodes for the rays, since this is used to form the file name.
use_low_dip_mask = True
use_dip_direction_mask = True

# map_file = map_folder + map_name + '.shp'
map_file = map_folder + '/' + map_name + '/' + map_name + '.shp'
rays_file = results_folder + str(nodes) + '_nodes_' + map_name + '_rays.shp'
segments_file = results_folder + str(nodes) + '_nodes_' + map_name + '_segments.shp'
match_lines_file = results_folder + str(nodes) + '_nodes_' + map_name + '_Match_lines.shp'
match_points_file = results_folder + str(nodes) + '_nodes_' + map_name + '_Match_points.shp'
cluster_points_file = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_points.shp'
cluster_lines_file = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_lines.shp'
cluster_points_file_preLOF = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_points_preLOF.shp'
cluster_lines_file_preLOF = results_folder + str(nodes) + '_nodes_' + map_name + '_cluster_lines_preLOF.shp'
fold_axes_file = results_folder + str(nodes) + '_nodes_' + map_name + '_fold_axes.shp'
fold_areas_file = results_folder + str(nodes) + '_nodes_' + map_name + '_fold_area.shp'

fold_type_colors = ['blue','red']
cluster_colors = ['blue','orange','red']

strat_num_shift = 1 #Shift down by this so the youngest unit is 1.
vmin = 1#
vmax = 7-strat_num_shift

#Define a function to plot the base map, since we will be doing this repeatedly.
def plot_map(map_file,legend=True):
    map_gdf = gp.GeoDataFrame.from_file(map_file, geom_col='geometry', crs='epsg:2154')
    map_gdf['RelAge'] -= strat_num_shift
    fig, ax = plt.subplots(1, 1)
    divider = make_axes_locatable(ax)
    if legend:
        cax = divider.append_axes("right", size="5%", pad=0.1)
    else:
        cax = None
    colormap = 'Greys'
    legend_kwds = {'orientation':'vertical',
                   'label':'Relative Age',}
    map_gdf.plot(ax=ax, column='RelAge', edgecolor='black',cmap=colormap,cax=cax,
             categorical=False,legend=legend,vmin=vmin,vmax=vmax,legend_kwds=legend_kwds)
    if legend:
        cax.invert_yaxis()
        cax.text(1.5, 0.93, 'Younger', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
        cax.text(1.5, 0.07, 'Older', horizontalalignment='left',verticalalignment='center',transform=cax.transAxes)
    ax.set_axis_off() #Don't show the axis numbers.
    ax.plot([1e4,1.1e4],[1e3,1e3],'k') #Scale bar
    ax.text(1e4,1.1e3,'1 km',horizontalalignment='left',verticalalignment='bottom',fontsize='small')
    return fig,ax

#Plot the map by itself.
fig1,ax1 = plot_map(map_file)

#Plot the map with one ray across it.
#I probably don't actually need this figure. Just the one after it should be enough.
plot_ray_ind = 39
fig2,ax2 = plot_map(map_file)
rays_gdf = gp.GeoDataFrame.from_file(rays_file, geom_col='geometry', crs='epsg:2154')
line = rays_gdf.geometry[plot_ray_ind]
x,y = line.xy
ax2.plot(x,y,'b-')

#Plot the map with the ray divided into segments.
fig3,ax3 = plot_map(map_file)
segments_gdf = gp.GeoDataFrame.from_file(segments_file, geom_col='geometry', crs='epsg:2154')
segs = segments_gdf[segments_gdf.index_righ==plot_ray_ind]
for i in segs.index:
    x,y = segs.geometry[i].xy
    # ax3.plot(x,y,'b.-')
    ax3.plot(x,y,'b-')
    for j in range(len(x)):
        ax3.plot([x[j],x[j]],[y[j]-200,y[j]+200],'b-')

#Plot a map with just the segments identified as containing possible folds.
match_lines_gdf = gp.GeoDataFrame.from_file(match_lines_file, geom_col='geometry', crs='epsg:2154')
fig4,ax4 = plot_map(map_file,legend=True)
for i in match_lines_gdf.index:
    x,y = match_lines_gdf.geometry[i].xy
    ax4.plot(x,y,'b-',linewidth=0.5)
    
#Plot the midpoints classified as anticlines or synclines.
fig5,ax5 = plot_map(map_file,legend=False)   
match_points_gdf = gp.GeoDataFrame.from_file(match_points_file, geom_col='geometry', crs='epsg:2154')
handles = [None,None]
for i,ftype in enumerate(['anticline','syncline']):
    gdf = match_points_gdf[match_points_gdf['fold_type']==ftype]
    x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
    y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
    handles[i] = ax5.plot(x,y,'.',color=fold_type_colors[i])[0]
ax5.legend(handles=handles,labels=['possible anticlines','possible synclines'])

#Plot the same thing, but with the ones that fail the topographic tests masked out.
fig6,ax6 = plot_map(map_file,legend=False)   
n1x,n1y,n1z = [match_points_gdf['n1x'].values,match_points_gdf['n1y'].values,match_points_gdf['n1z'].values]
n2x,n2y,n2z = [match_points_gdf['n2x'].values,match_points_gdf['n2y'].values,match_points_gdf['n2z'].values]
nh1 = np.sqrt(n1x**2.+n1y**2.)
nh2 = np.sqrt(n2x**2.+n2y**2.)
phi1 = np.arctan(np.abs(n1z)/nh1) #Plunge
phi2 = np.arctan(np.abs(n2z)/nh2) #Plunge
dip1 = 90-phi1*180.0/np.pi
dip2 = 90-phi2*180.0/np.pi
mask = np.ones(phi1.shape,dtype=bool)
if use_low_dip_mask:
    mask = mask & (dip1>5) & (dip2>5)
dip_direction_mask = ((match_points_gdf['fold_type']=='anticline') & (match_points_gdf['DipDirs']=='apart')) | (
    (match_points_gdf['fold_type']=='syncline') & (match_points_gdf['DipDirs']=='together'))
if use_dip_direction_mask:
    mask = mask & dip_direction_mask #Use this.
handles = [None,None,None]
gdf = match_points_gdf[~mask]
x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
handles[2] = ax6.plot(x,y,'.',color='black',alpha=0.1)[0]
for i,ftype in enumerate(['anticline','syncline']):
    gdf = match_points_gdf[(match_points_gdf['fold_type']==ftype) & mask]
    x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
    y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
    handles[i] = ax6.plot(x,y,'.',color=fold_type_colors[i])[0]
ax6.legend(handles=handles,labels=['possible anticlines','possible synclines','rejected'])

#Plot the clustered midpoints.
cluster_points_gdf = gp.GeoDataFrame.from_file(cluster_points_file, geom_col='geometry', crs='epsg:2154')
gdf_sans_noise_points = cluster_points_gdf[cluster_points_gdf['cluster'] != -1]
noise = cluster_points_gdf[cluster_points_gdf['cluster'] == -1]
fig7,ax7 = plot_map(map_file,legend=False)
clusters_unique = pd.unique(gdf_sans_noise_points['cluster']).tolist()
# NUM_COLORS = len(clusters_unique)
# colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, NUM_COLORS)]
handles = [None,None,None,None]
if not noise.empty:
    x = [noise.geometry[i].xy[0][0] for i in noise.index]
    y = [noise.geometry[i].xy[1][0] for i in noise.index]
    handles[3] = ax7.plot(x,y,'.',color='black',alpha=0.1)[0]
i = 0
for color, (index, group), c in zip(cluster_colors[0:len(clusters_unique)], gdf_sans_noise_points.groupby(['cluster']), clusters_unique):
    x = [group.geometry[i].xy[0][0] for i in group.index]
    y = [group.geometry[i].xy[1][0] for i in group.index]
    handles[i] = ax7.plot(x,y,'.',color=cluster_colors[i])[0]
    i += 1
ax7.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3','not clustered'],ncols=2)

#Plot the clustered line segments.
cluster_lines_gdf = gp.GeoDataFrame.from_file(cluster_lines_file, geom_col='geometry', crs='epsg:2154')
gdf_sans_noise_lines = cluster_lines_gdf[cluster_lines_gdf['cluster'] != -1]
noise = cluster_lines_gdf[cluster_lines_gdf['cluster'] == -1]
fig8,ax8 = plot_map(map_file,legend=False)
clusters_unique = pd.unique(gdf_sans_noise_lines['cluster']).tolist()
handles = [None,None,None,None]
if not noise.empty:
    for j in noise.index:
        x,y = noise.geometry[j].xy
        if j==noise.index[0]:
            handles[3] = ax8.plot(x,y,'-',color='black',alpha=0.1)[0]
        else:
            ax8.plot(x,y,'-',color='black',alpha=0.1)
i = 0
for color, (index, group), c in zip(cluster_colors[0:len(clusters_unique)], gdf_sans_noise_lines.groupby(['cluster']), clusters_unique):
    for j in group.index:
        x,y = group.geometry[j].xy
        if j==group.index[0]:
            handles[i] = ax8.plot(x,y,'-',color=cluster_colors[i])[0]
        else:
            ax8.plot(x,y,'-',color=cluster_colors[i])
    i += 1
ax8.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3','not clustered'],ncols=2)

#Plot the midpoints that will be used for determining the axes.
fig9,ax9 = plot_map(map_file,legend=False)
handles = [None,None,None]
for i,c in enumerate(clusters_unique):
    gdf = gdf_sans_noise_points[(gdf_sans_noise_points['cluster']==c) & (gdf_sans_noise_points['max_angle']<60)]
    x = [gdf.geometry[i].xy[0][0] for i in gdf.index]
    y = [gdf.geometry[i].xy[1][0] for i in gdf.index]
    handles[i] = ax9.plot(x,y,'.',color=cluster_colors[i])[0]
ax9.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3'],ncols=2)

#Plot the fold axes.
fig10,ax10 = plot_map(map_file,legend=True)
axes_gdf = gp.GeoDataFrame.from_file(fold_axes_file, geom_col='geometry', crs='epsg:2154')
for i,c in enumerate(clusters_unique):
    ind = axes_gdf.index[axes_gdf.cluster==c][0] #This should only return one item.
    x,y = axes_gdf.geometry[ind].xy
    ax10.plot(x,y,'-',color=cluster_colors[i])

#Plot the line segments that will be used for determining the fold areas.
cluster_lines_gdf = gp.GeoDataFrame.from_file(cluster_lines_file, geom_col='geometry', crs='epsg:2154')
fig11,ax11 = plot_map(map_file,legend=False)
handles = [None,None,None]
for i,c in enumerate(clusters_unique):
    gdf = cluster_lines_gdf[cluster_lines_gdf['cluster']==c]
    for j,ind in enumerate(gdf.index):
        x,y = gdf.geometry[ind].xy
        h = ax11.plot(x,y,color=cluster_colors[i],linewidth=0.5)[0]
        if j == 1:
            handles[i] = h
ax11.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3'],ncols=2)

#Plot the fold areas.
fig12,ax12 = plot_map(map_file,legend=True)
areas_gdf = gp.GeoDataFrame.from_file(fold_areas_file, geom_col='geometry', crs='epsg:2154')
newcmp = ListedColormap(cluster_colors)
areas_gdf.plot(ax=ax12,categorical=True,cmap=newcmp,legend=False, alpha=0.5, edgecolor='black')

#Plot the clustered midpoints before outlier removal.
cluster_points_gdf_preLOF = gp.GeoDataFrame.from_file(cluster_points_file_preLOF, geom_col='geometry', crs='epsg:2154')
gdf_sans_noise_points = cluster_points_gdf_preLOF[cluster_points_gdf_preLOF['cluster'] != -1]
noise = cluster_points_gdf_preLOF[cluster_points_gdf_preLOF['cluster'] == -1]
fig13,ax13 = plot_map(map_file,legend=False)
clusters_unique = pd.unique(gdf_sans_noise_points['cluster']).tolist()
# NUM_COLORS = len(clusters_unique)
# colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, NUM_COLORS)]
handles = [None,None,None,None]
if not noise.empty:
    x = [noise.geometry[i].xy[0][0] for i in noise.index]
    y = [noise.geometry[i].xy[1][0] for i in noise.index]
    handles[3] = ax13.plot(x,y,'.',color='black',alpha=0.1)[0]
i = 0
for color, (index, group), c in zip(cluster_colors[0:len(clusters_unique)], gdf_sans_noise_points.groupby(['cluster']), clusters_unique):
    x = [group.geometry[i].xy[0][0] for i in group.index]
    y = [group.geometry[i].xy[1][0] for i in group.index]
    handles[i] = ax13.plot(x,y,'.',color=cluster_colors[i])[0]
    i += 1
ax13.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3','not clustered'],ncols=2)

#Plot the clustered line segments before outlier removal.
cluster_lines_gdf_preLOF = gp.GeoDataFrame.from_file(cluster_lines_file_preLOF, geom_col='geometry', crs='epsg:2154')
gdf_sans_noise_lines = cluster_lines_gdf_preLOF[cluster_lines_gdf_preLOF['cluster'] != -1]
noise = cluster_lines_gdf_preLOF[cluster_lines_gdf_preLOF['cluster'] == -1]
fig14,ax14 = plot_map(map_file,legend=False)
clusters_unique = pd.unique(gdf_sans_noise_lines['cluster']).tolist()
handles = [None,None,None,None]
if not noise.empty:
    for j in noise.index:
        x,y = noise.geometry[j].xy
        if j==noise.index[0]:
            handles[3] = ax14.plot(x,y,'-',color='black',alpha=0.1)[0]
        else:
            ax14.plot(x,y,'-',color='black',alpha=0.1)
i = 0
for color, (index, group), c in zip(cluster_colors[0:len(clusters_unique)], gdf_sans_noise_lines.groupby(['cluster']), clusters_unique):
    for j in group.index:
        x,y = group.geometry[j].xy
        if j==group.index[0]:
            handles[i] = ax14.plot(x,y,'-',color=cluster_colors[i])[0]
        else:
            ax14.plot(x,y,'-',color=cluster_colors[i])
    i += 1
ax14.legend(handles=handles,labels=['cluster 1','cluster 2','cluster 3','not clustered'],ncols=2)