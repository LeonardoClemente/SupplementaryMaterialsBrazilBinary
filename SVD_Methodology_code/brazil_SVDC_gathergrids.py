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
                   (0.70, .4, .4),
                   (0.799001, .7, 7),
                   (.840, c2, c2),
                   (.840001, 0, 0),
                   (1.0,  0, 0)),

         'green': ((0.0,  c1, c1),
                   (0.70, .4, .4),
                   (0.799001, .7, 7),
                   (.840, c2, c2),
                   (.840001, 0, 0),
                   (1.0,  0, 0)),

         'blue':  ((0.0, c1, c1),
                   (0.70, .4, .4),
                   (0.799001, .7, 7),
                   (.84, c2, c2),
                   (.840001, c3, c3),
                   (1.0,  1.0, 1.0))}


custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
index_dates = ['2000-06-01', '2000-07-01', '2000-08-01','2000-09-01', '2000-10-01',\
               '2000-11-01' ,'2000-12-01', '2001-01-01', '2001-02-01']
## END CUSTOM COLORMAP
brazil_stations = ['ARACAJU', 'BELO HORIZONTE', 'MANAUS', 'RECIFE (CURADO)']
 #'Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2'
station =  'Manaus_MERRA_2'


data = pickle.load(open( "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS/SVDC_{0}_all_grids.p".format(station), "rb" ))
data2017 = pickle.load(open( "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS/SVDC_{0}2017.p".format(station), "rb" ))
distance_grids = data[0]
grid2017 = data2017[0]
distance_grids.append(grid2017)
distance_grid = np.zeros_like(distance_grids[0])
t0_vector = data[1]
p_vector = data[2]

print(len(distance_grids[5:]))
for dg in distance_grids[5:]:
    distance_grid += dg
    #im = plt.matshow(dg, cmap=plt.cm.hot, aspect='auto', origin='lower') # pl is pylab imported a pl
    #plt.colorbar()
    #plt.show()


index_dates_ = ['JUN-01', 'JUL-01', 'AGO-01','SEP-01', 'OCT-01',\
               'NOV-01' ,'DEC-01', 'JAN-01', 'FEB-01']
index_values = []
year = list(range(2012,2018))
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

im = plt.matshow(distance_grid, cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=6) # pl is pylab imported a pl plt.cm.hot

plt.colorbar()
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,index_dates_, rotation='90')
plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
#plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=10)
plt.ylabel('P')
plt.xlabel('2012-2017 (6 years)'.format(station))
plt.xlim([index_values[0], index_values[4]-3])
#Choose highest modes
fig=plt.gcf()
fig.set_size_inches([11,8])
fig.savefig('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/plots/compare_techniques/{0}/accumulated.png'.format(station), dpi=300)
plt.close()


for i, dg in enumerate(distance_grids[5:]):
    im = plt.matshow(dg, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=1) # pl is pylab imported a pl plt.cm.hot
    plt.colorbar()
    #plt.tick_params(axis='x',which='minor',bottom='off')
    plt.xticks(index_values,index_dates_, rotation='90')
    plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
    #plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=10)
    plt.ylabel('P')
    plt.xlabel('{0}'.format(year[i]))
    plt.xlim([index_values[0], index_values[4]-3])
    #Choose highest modes
    fig=plt.gcf()
    fig.set_size_inches([11,8])
    fig.savefig('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/plots/compare_techniques/{0}/{1}.png'.format(station,year[i]), dpi=300)
