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
import pickle
from matplotlib.colors import LinearSegmentedColormap

data = pickle.load(open( "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS/SVM_linear_norm.p", "rb" ))
kernel = 'linear'
distance_grid = data[0]
t0_vector = data[1]
p_vector = data[2]


if kernel == 'rbf':
        c1=0
        c2=.6
        c3=1

        cdict = {'red':   ((0.0,  c1, c1),
                           (0.95, c2, c2),
                           (0.950001, c3, c3),
                           (1.0,  1.0, 1.0)),

                 'green': ((0.0,  c1, c1),
                           (0.95, c2, c2),
                           (0.950001, c3, c3),
                           (1.0,  1.0, 1.0)),

                 'blue':  ((0.0, c1, c1),
                           (0.95, c2, c2),
                           (0.950001, c3, c3),
                           (1.0,  1.0, 1.0))}

elif kernel == 'linear':

    cdict = {'red':   ((0.0,  0.0, 0.0),
                       (0.79999, 0.0, 0.0),
                       (0.8, 0.8, 0.8),
                       (1.0,  1.0, 1.0)),

             'green': ((0.0,  0.0, 0.0),
                       (0.79999, 0.0, 0.0),
                       (0.8, 0.6, 0.6),
                       (1.0,  1.0, 1.0)),

             'blue':  ((0.0,  0.2, 0.2),
                       (0.79999, 1.0, 1.0),
                       (0.80000001, 0,0),
                       (1.0,  0, 0))}

custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
index_dates = ['2001-06-01', '2001-07-01', '2001-08-01','2001-09-01', '2001-10-01',\
               '2001-11-01' ,'2001-12-01', '2002-01-01', '2002-02-01']
index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

im = plt.matshow(distance_grid, cmap=custom_cmap, origin='lower', vmax=1, vmin=0) #aspect='auto' # pl is pylab imported a pl




plt.colorbar()
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,index_dates, rotation='90')
plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
#plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=10)
plt.ylabel('P')
plt.xlabel('Start date.')
plt.show()
#Choose highest modes
