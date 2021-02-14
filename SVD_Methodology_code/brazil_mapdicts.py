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
from matplotlib.ticker import MaxNLocator,IndexFormatter
import matplotlib.ticker as mticker



## END CUSTOM COLORMAP
brazil_stations = ['ARACAJU', 'BELO HORIZONTE', 'MANAUS', 'RECIFE (CURADO)']
 #'Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2'
station = 'BeloHorizonte_MERRA_2'# 'Manaus_MERRA_2' #'BeloHorizonte_MERRA_2'

heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_dicts'
output_folder ='/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots'
proc_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots' + '/{0}/compareProcessed'.format(station)
mod_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots' + '/{0}/compareMod'.format(station)
sy=2006
ey=2018

'''
We need to get the heatmaps for SVM v0 and tree v2
'''
svm_dict = {}
version_svm = 6

forest_dict  ={}
version_forest = 6

for year in range(sy, ey):
    data = pickle.load(open( "{2}/grid_dict_{1}_{0}_second.p".format(year, station, heatmap_path), "rb" ))
    grid_dict = data[0]
    t0_vector = data[1]
    p_vector = data[2]

    for proc in ['unprocessed']:

        for version in range(10):

            for mod in ['svm', 'forest']:
                heatmap = grid_dict[proc][version][mod]
                heatmap[heatmap==-1] = 0

                if mod == 'svm' and proc == 'unprocessed' and version == version_svm:
                    svm_dict[year] = heatmap
                if mod == 'forest' and proc == 'unprocessed' and version == version_forest:
                    forest_dict[year] = heatmap

pickle.dump([svm_dict, t0_vector, p_vector], open(heatmap_path+'/{0}_svm_{1}.p'.format(station, version_svm), 'wb'))
pickle.dump([forest_dict, t0_vector, p_vector], open(heatmap_path+'/{0}_forest_{1}.p'.format(station, version_forest), 'wb'))
