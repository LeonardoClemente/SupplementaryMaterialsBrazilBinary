import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import copy
import time
from sklearn.utils.extmath import randomized_svd as svd
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from scipy.spatial import distance
from matplotlib.colors import LinearSegmentedColormap
import pickle



### CUSTOM COLORMAP
c1=0
c2=.7
c3=1

cdict = {'red':   ((0.0,  c1, c1),
                   (0.68, .4, .4),
                   (0.680001, .7, 7),
                   (.84, c2, c2),
                   (.84001, 0, 0),
                   (1.0,  0, 0)),

         'green': ((0.0,  c1, c1),
                   (0.68, .4, .4),
                   (0.680001, .7, 7),
                   (.84, c2, c2),
                   (.84001, 0, 0),
                   (1.0,  0, 0)),

         'blue':  ((0.0, c1, c1),
                   (0.68, .4, .4),
                   (0.680001, .7, 7),
                   (.84, c2, c2),
                   (.84001, c3, c3),
                   (1.0,  1.0, 1.0))}


custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
index_dates = ['2001-06-01', '2001-07-01', '2001-08-01','2001-09-01', '2001-10-01',\
               '2001-11-01' ,'2001-12-01', '2002-01-01', '2002-02-01']

## END CUSTOM COLORMAP
brazil_stations = ['ARACAJU', 'BELO HORIZONTE', 'MANAUS', 'RECIFE (CURADO)']
 #'Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2'
station =  'MANAUS'
station2 = 'Manaus_MERRA_2'

data = pickle.load(open( "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS/SVDC_{0}.p".format(station), "rb" ))
data2 = pickle.load(open( "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS/SVDC_{0}_all_grids.p".format(station2), "rb" ))



distance_grid = data[0]



distance_grid2 = data2[0]
t0_vector = data[1]
p_vector = data[2]

ax = plt.subplot(2,1,1)
index_dates_ = ['JUN', 'JUL', 'AGO','SEP', 'OCT',\
               'NOV' ,'DEC', 'JAN', 'FEB']
index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

im = ax.matshow(distance_grid, cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=6) # pl is pylab imported a pl plt.cm.hot

#plt.colorbar(im, cax=ax)
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,['','','','','','','','',''], rotation='90')
ax.set_yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
#plt.locator_params(axis='x', nbins=20)
ax.locator_params(axis='y', nbins=10)
ax.yaxis.tick_right()
plt.ylabel('P')
plt.xlabel(station)



#-----------------------------------

ax2 = plt.subplot(2,1,2)

index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

im2 = ax2.matshow(distance_grid2, cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=6) # pl is pylab imported a pl plt.cm.hot

#plt.colorbar(im2, cax=ax2)
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,index_dates_, rotation='90')
ax2.set_yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
#plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=10)
plt.ylabel('P')
plt.xlabel(station2)



plt.show()
#Choose highest modes
