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


### CUSTOM COLORMAP
c1=0
c2=.7
c3=1
sy = 2006
ey = 2013
v= 4
cdict = {'red':   ((0.0,  c1, c1),
                   (0.80000, .6, .6),
                   (0.80001, 0, 0),
                   (.9500, 0, 0),
                   (.9501, 1, 1),
                   (1.0,  1, 1)),

         'green': ((0.0,  c1, c1),
                   (0.80000, .6, .6),
                   (0.80001, 0, 0),
                   (.9500, 0, 0),
                   (.9501, 1, 1),
                   (1.0,  1, 1)),

         'blue':  ((0.0, c1, c1),
                   (0.80000, .6, .6),
                   (0.80001, 1, 1),
                   (.9500, 1, 1),
                   (.9501, 1, 1),
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
proc_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots' + '/{0}/compareProcessed'.format(station)
mod_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots' + '/{0}/compareMod'.format(station)



# Compare processing vs unprocessed
'''
To compare the processed vs unprocessed

FOR A YEAR:  we plot in a n_versions*2

where we plot the processed for each year

'''

acc_heatmaps = {}

for year in range(sy, ey):
    fig, axarr = plt.subplots(12,2)
    data = pickle.load(open( "{2}/grid_dict_{1}_{0}.p".format(year, station, heatmap_path), "rb" ))
    grid_dict = data[0]
    t0_vector = data[1]
    p_vector = data[2]

    # Get indices for monthly x ticks
    index_dates_ = ['JUN-01', 'JUL-01', 'AGO-01','SEP-01', 'OCT-01',\
                   'NOV-01' ,'DEC-01', 'JAN-01', 'FEB-01']
    index_values = []
    for d in index_dates:
        index_values.append(t0_vector.index(d))

    for proc in ['unprocessed', 'processed']:

        if proc == 'unprocessed':
            n_proc = 0
        else:
            n_proc = 1

        for version in range(0,6):
            for mod in ['forest', 'svm']:
                if mod == 'forest':
                    n_mod = 0
                else:
                    n_mod = 1
                heatmap = grid_dict[proc][version][mod]
                im = axarr[version+6*n_mod, n_proc].matshow(heatmap, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=1) # pl is pylab imported a pl plt.cm.hot
                if version < 5:
                    axarr[version+6*n_mod, n_proc].axes.get_xaxis().set_visible(False)
                else:
                    axarr[version+6*n_mod, n_proc].xaxis.tick_bottom()
                #plt.colorbar()
                #plt.tick_params(axis='x',which='minor',bottom='off')
                plt.xticks(index_values,index_dates_, rotation='90')
                #plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
                #plt.locator_params(axis='x', nbins=20)
                #plt.locator_params(axis='y', nbins=10)
                plt.ylabel('P')
                plt.xlabel(station)


                if version==0 and n_mod==0:
                    axarr[version+6*n_mod, n_proc].set_title('{0}'.format(proc))
                if n_proc == 1 and version+6*n_mod == 11:
                    plt.xticks(index_values,index_dates_, rotation='90')
                    plt.xlim([index_values[0], index_values[4]])
                else:
                    axarr[version+6*n_mod, n_proc].set_xticks(list(range(len(t0_vector))), minor=False)
                    axarr[version+6*n_mod, n_proc].xaxis.set_major_formatter(IndexFormatter([date[5:] for date in t0_vector]))
                    axarr[version+6*n_mod, n_proc].xaxis.set_major_locator(mticker.MaxNLocator(5))
                    axarr[version+6*n_mod, n_proc].set_xlim([index_values[0], index_values[4]])
                if n_proc == 0:
                    axarr[version+6*n_mod, n_proc].set_ylabel('{0} version {1}'.format(mod, version))
                heatmap[heatmap == -1] = 0

                if year == sy:
                    acc_heatmaps[proc+str(version)+mod] = heatmap
                else:
                    acc_heatmaps[proc+str(version)+mod] += heatmap
    fig = plt.gcf()
    fig.set_size_inches(6, 20)
    plt.savefig('{0}/{1}.png'.format(proc_folder, year), format='png', dpi=300)
    plt.close()

fig, axarr = plt.subplots(12,2)

for proc in ['unprocessed', 'processed']:

    if proc == 'unprocessed':
        n_proc = 0
    else:
        n_proc = 1

    for version in range(0,6):
        for mod in ['forest', 'svm']:
            if mod == 'forest':
                n_mod = 0
            else:
                n_mod = 1

            heatmap = acc_heatmaps[proc+str(version)+mod]
            im = axarr[version+6*n_mod, n_proc].matshow(heatmap,cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=ey-sy) # pl is pylab imported a pl plt.cm.hot
            if version < 5:
                axarr[version+6*n_mod, n_proc].axes.get_xaxis().set_visible(False)
            else:
                axarr[version+6*n_mod, n_proc].xaxis.tick_bottom()
            #plt.colorbar()
            #plt.tick_params(axis='x',which='minor',bottom='off')
            plt.xticks(index_values,index_dates_, rotation='90')
            #plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
            #plt.locator_params(axis='x', nbins=20)
            #plt.locator_params(axis='y', nbins=10)
            plt.ylabel('P')
            plt.xlabel(station)


            if version==0 and n_mod==0:
                axarr[version+6*n_mod, n_proc].set_title('{0}'.format(proc))
            if n_proc == 1 and version+6*n_mod == 11:
                plt.xticks(index_values,index_dates_, rotation='90')
                plt.xlim([index_values[0], index_values[4]])
            else:
                axarr[version+6*n_mod, n_proc].set_xticks(list(range(len(t0_vector))), minor=False)
                axarr[version+6*n_mod, n_proc].xaxis.set_major_formatter(IndexFormatter([date[5:] for date in t0_vector]))
                axarr[version+6*n_mod, n_proc].xaxis.set_major_locator(mticker.MaxNLocator(5))
                axarr[version+6*n_mod, n_proc].set_xlim([index_values[0], index_values[4]])
            if n_proc == 0:
                axarr[version+6*n_mod, n_proc].set_ylabel('{0} version {1}'.format(mod, version))
            heatmap[heatmap == -1] = 0

fig = plt.gcf()
fig.set_size_inches(6, 20)
plt.savefig('{0}/stacked.png'.format(proc_folder), format='png', dpi=300)
plt.close()


'''
COMPARING TREE VS SVM

'''


proc_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_plots' + '/{0}/compareProcessed'.format(station)
acc_heatmaps = {}

for year in range(sy, ey):
    fig, axarr = plt.subplots(12,2)
    data = pickle.load(open( "{2}/grid_dict_{1}_{0}.p".format(year, station, heatmap_path), "rb" ))
    grid_dict = data[0]
    t0_vector = data[1]
    p_vector = data[2]

    # Get indices for monthly x ticks
    index_dates_ = ['JUN-01', 'JUL-01', 'AGO-01','SEP-01', 'OCT-01',\
                   'NOV-01' ,'DEC-01', 'JAN-01', 'FEB-01']
    index_values = []
    for d in index_dates:
        index_values.append(t0_vector.index(d))

    for proc in ['unprocessed', 'processed']:

        if proc == 'unprocessed':
            n_proc = 0
        else:
            n_proc = 1

        for version in range(0,6):
            for mod in ['forest', 'svm']:
                if mod == 'forest':
                    n_mod = 0
                else:
                    n_mod = 1
                heatmap = grid_dict[proc][version][mod]
                im = axarr[version+6*n_proc, n_mod].matshow(heatmap, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=1) # pl is pylab imported a pl plt.cm.hot
                if version < 5:
                    axarr[version+6*n_proc, n_mod].axes.get_xaxis().set_visible(False)
                else:
                    axarr[version+6*n_proc, n_mod].xaxis.tick_bottom()
                #plt.colorbar()
                #plt.tick_params(axis='x',which='minor',bottom='off')
                plt.xticks(index_values,index_dates_, rotation='90')
                #plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
                #plt.locator_params(axis='x', nbins=20)
                #plt.locator_params(axis='y', nbins=10)
                plt.ylabel('P')
                plt.xlabel(station)


                if version==0 and n_proc == 0:
                    axarr[version+6*n_proc, n_mod].set_title('{0}'.format(mod))
                if n_mod == 1 and version+6*n_proc == 11:
                    plt.xticks(index_values,index_dates_, rotation='90')
                    plt.xlim([index_values[0], index_values[4]])
                else:
                    axarr[version+6*n_proc, n_mod].set_xticks(list(range(len(t0_vector))), minor=False)
                    axarr[version+6*n_proc, n_mod].xaxis.set_major_formatter(IndexFormatter([date[5:] for date in t0_vector]))
                    axarr[version+6*n_proc, n_mod].xaxis.set_major_locator(mticker.MaxNLocator(5))
                    axarr[version+6*n_proc, n_mod].set_xlim([index_values[0], index_values[4]])
                if n_mod == 0:
                    axarr[version+6*n_proc, n_mod].set_ylabel('{0} version {1}'.format(proc, version))
                heatmap[heatmap == -1] = 0

                if year == sy:
                    acc_heatmaps[proc+str(version)+mod] = heatmap
                else:
                    acc_heatmaps[proc+str(version)+mod] += heatmap
    fig = plt.gcf()
    fig.set_size_inches(6, 20)
    plt.savefig('{0}/{1}.png'.format(mod_folder, year), format='png', dpi=300)
    plt.close()

fig, axarr = plt.subplots(12,2)

for proc in ['unprocessed', 'processed']:

    if proc == 'unprocessed':
        n_proc = 0
    else:
        n_proc = 1

    for version in range(0,6):
        for mod in ['forest', 'svm']:
            if mod == 'forest':
                n_mod = 0
            else:
                n_mod = 1

            heatmap = acc_heatmaps[proc+str(version)+mod]
            im = axarr[version+6*n_proc, n_mod].matshow(heatmap,cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=ey-sy) # pl is pylab imported a pl plt.cm.hot
            if version < 5:
                axarr[version+6*n_proc, n_mod].axes.get_xaxis().set_visible(False)
            else:
                axarr[version+6*n_proc, n_mod].xaxis.tick_bottom()
            #plt.colorbar()
            #plt.tick_params(axis='x',which='minor',bottom='off')
            plt.xticks(index_values,index_dates_, rotation='90')
            #plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
            #plt.locator_params(axis='x', nbins=20)
            #plt.locator_params(axis='y', nbins=10)
            plt.ylabel('P')
            plt.xlabel(station)


            if version==0 and n_proc == 0:
                axarr[version+6*n_proc, n_mod].set_title('{0}'.format(mod))
            if n_mod == 1 and version+6*n_proc == 11:
                plt.xticks(index_values,index_dates_, rotation='90')
                plt.xlim([index_values[0], index_values[4]])
            else:
                axarr[version+6*n_proc, n_mod].set_xticks(list(range(len(t0_vector))), minor=False)
                axarr[version+6*n_proc, n_mod].xaxis.set_major_formatter(IndexFormatter([date[5:] for date in t0_vector]))
                axarr[version+6*n_proc, n_mod].xaxis.set_major_locator(mticker.MaxNLocator(5))
                axarr[version+6*n_proc, n_mod].set_xlim([index_values[0], index_values[4]])
            if n_mod == 0:
                axarr[version+6*n_proc, n_mod].set_ylabel('{0} ver {1}'.format(proc[:5], version))
            heatmap[heatmap == -1] = 0

fig = plt.gcf()
fig.set_size_inches(6, 20)
plt.savefig('{0}/stacked.png'.format(mod_folder), format='png', dpi=300)
plt.close()

# ----------------

'''
acc_heatmaps = {}
for year in range(sy, ey):
    data = pickle.load(open( "{2}/grid_dict_{1}_{0}.p".format(year, station, heatmap_path), "rb" ))
    grid_dict = data[0]
    t0_vector = data[1]
    p_vector = data[2]

    # Get indices for monthly x ticks
    index_dates_ = ['JUN-01', 'JUL-01', 'AGO-01','SEP-01', 'OCT-01',\
                   'NOV-01' ,'DEC-01', 'JAN-01', 'FEB-01']
    index_values = []
    for d in index_dates:
        index_values.append(t0_vector.index(d))

    for proc in ['unprocessed', 'processed']:
        for version in range(0,6):
            for mod in ['forest', 'svm']:
                heatmap = grid_dict[proc][version][mod]
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
                plt.savefig('{0}/{1}/{2}/{4}_{5}_{3}.png'.format(output_folder,station,year, version, proc, mod), format='png', dpi=300)
                plt.close()
                heatmap[heatmap == -1] = 0

                if year == sy:
                    acc_heatmaps[proc+str(version)+mod] = heatmap
                else:
                    acc_heatmaps[proc+str(version)+mod] += heatmap


for k, heatmap in acc_heatmaps.items():
    im = plt.matshow(heatmap, cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=ey-sy) # pl is pylab imported a pl plt.cm.hot
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
    plt.savefig('{0}/{1}/stacked_{2}.png'.format(output_folder,station,k), format='png', dpi=300)
    plt.close()
'''
