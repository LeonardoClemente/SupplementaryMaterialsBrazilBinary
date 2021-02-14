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


data = pickle.load(open( "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS/modes12.p", "rb" ))

distance_grid = data[0]
t0_vector = data[1]
p_vector = data[2]


index_dates = ['2001-06-01', '2001-07-01', '2001-08-01','2001-09-01', '2001-10-01',\
               '2001-11-01' ,'2001-12-01', '2002-01-01', '2002-02-01']
index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

im = plt.matshow(distance_grid, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=6) # pl is pylab imported a pl

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
