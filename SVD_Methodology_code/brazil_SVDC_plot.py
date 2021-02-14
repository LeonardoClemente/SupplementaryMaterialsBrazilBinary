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
sy = 2006
ey = 2010
v= 4
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
index_dates = ['2000-06-01', '2000-07-01', '2000-08-01','2000-09-01', '2000-10-01',\
               '2000-11-01' ,'2000-12-01', '2001-01-01', '2001-02-01']

## END CUSTOM COLORMAP
brazil_stations = ['ARACAJU', 'BELO HORIZONTE', 'MANAUS', 'RECIFE (CURADO)']
 #'Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2'
station =  'BeloHorizonte_MERRA_2'

heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_dicts'
output_folder ='/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots'


data = pickle.load(open( "{1}/v{4}_{0}_{2}_{3}.p".format(station, heatmap_path, sy, ey, v), "rb" ))

heatmap_dict = data[0]
t0_vector = data[1]
p_vector = data[2]
accumulated_heatmap = np.zeros_like(heatmap_dict[2006])

# Get indices for monthly x ticks
index_dates_ = ['JUN-01', 'JUL-01', 'AGO-01','SEP-01', 'OCT-01',\
               'NOV-01' ,'DEC-01', 'JAN-01', 'FEB-01']
index_values = []
for d in index_dates:
    index_values.append(t0_vector.index(d))

for year, heatmap in heatmap_dict.items():
    accumulated_heatmap += heatmap
    im = plt.matshow(heatmap, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=1) # pl is pylab imported a pl plt.cm.hot
    plt.colorbar()
    #plt.tick_params(axis='x',which='minor',bottom='off')
    plt.xticks(index_values,index_dates_, rotation='90')
    plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
    #plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=10)
    plt.ylabel('P')
    plt.xlabel(station)
    plt.xlim([index_values[0], index_values[4]])
    fig = plt.gcf()
    fig.set_size_inches(13, 10)
    plt.savefig('{0}/{1}/{2}_{5}_{3}_{4}.png'.format(output_folder,station,year, sy, ey, v), format='png', dpi=300)
    plt.close()
#Choose highest modes


im = plt.matshow(accumulated_heatmap, cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=12) # pl is pylab imported a pl plt.cm.hot
plt.colorbar()
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,index_dates_, rotation='90')
plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
#plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=10)
plt.ylabel('P')
plt.xlabel(station)
plt.xlim([index_values[0], index_values[4]])
fig = plt.gcf()
fig.set_size_inches(13, 10)
plt.savefig('{0}/{1}/{4}_stacked_{2}_{3}.png'.format(output_folder,station,sy,ey, v), format='png', dpi=300)
