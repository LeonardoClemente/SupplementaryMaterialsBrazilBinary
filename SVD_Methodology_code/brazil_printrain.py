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
from detect_peaks import detect_peaks
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from brazil_functions import *
from brazil_constants import *
from progress.bar import Bar
import scipy
brazil_stations =  ['Aracaju', 'Sertaozinho', 'Rio', 'JuazeirodoNorte', 'SaoVicente', 'Parnaiba', 'Barretos', 'SaoGoncalo', 'Manaus', 'JiParana', 'Eunapolis', 'TresLagoas', 'SaoLuis', 'Guaruja', 'SantaCruz', 'Barueri', 'Rondonopolis', 'Maranguape', 'BarraMansa', 'BeloHorizonte']
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/SVDC_DATA_allvars'



def rmv_values(df, ignore = [.5], verbose = True):
    n_values = np.shape(df)[0]
    df = df[ df[TARGET].notnull() ]
    rem_nans = n_values - np.shape(df)[0]

    for i, val in enumerate(ignore):
        df = df[df[TARGET] != val]

    rem_vals =  n_values  - np.shape(df)[0] - rem_nans

    n_values_final = np.shape(df)[0]

    if verbose == True:
        print('Removed {0} NANs and {1} specified values from list :{2} \n'.format(rem_nans, rem_vals,ignore))
        print('Number of values before = {0}'.format(n_values))
        print('Number of values after =  {0}'.format(n_values_final))

    return df




years = [2000]
months = list(range(6, 13))
days = list(range(1, 32))

t0_vector = generate_date_vector(years, months, days) + generate_date_vector([2001], list(range(1,3)), days)
p_vector = [10, 50, 90]

index_dates = ['2000-06-01', '2000-07-01', '2000-08-01','2000-09-01', '2000-10-01',\
               '2000-11-01' ,'2000-12-01', '2001-01-01', '2001-02-01']
non_existent_dates = ['2001-02-29', '2001-02-30', '2001-02-31', '2000-06-31','2000-09-31', '2000-11-31',]
index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

 # Removing unexistent dates
for d in non_existent_dates:
    t0_vector.remove(d)



years = range(2000,2013)
t0 = '-06-01'
p = 100
TARGET = 'runoff_diff'
n_moments =  4
fig,ax = plt.subplots()


# Getting year histograms for a certain period
for i, station in enumerate(brazil_stations):
    epidemic_classification_dict = dict(zip([y for y in range(2001,2018)],EPIDEMIC_CLASSIFICATION_PER_STATION[station]))

    df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
    df = df.interpolate() #interpolating missing data
    df.index = pd.to_datetime(df.index)
    df.sort_index()
    epidemic_timeseries = []
    non_epidemic_timeseries = []
    for year in years:
        t = str(year) + t0
        ind = df.index.get_loc(t)

        if epidemic_classification_dict[int(year)+1] == 1:
            epidemic_timeseries.append(df[ind:ind+p][TARGET].values.ravel())
        else:
            non_epidemic_timeseries.append(df[ind:ind+p][TARGET].values.ravel())



    epidemic_timeseries = np.hstack(epidemic_timeseries)
    non_epidemic_timeseries = np.hstack(non_epidemic_timeseries)
    all_timeseries = np.hstack([epidemic_timeseries, non_epidemic_timeseries])
    mus = []
    stds = []
    skews = []
    kurtosis_ = []
    moments = []
    '''
    mu = float('%.2f'%(np.mean(target_values)))
    std = float('%.2f'%(np.std(target_values)))
    skew = float('%.2f'%(scipy.stats.skew(target_values)))
    kurtosis = float('%.2f'%(scipy.stats.kurtosis(target_values)))
    '''
    n_bins = 40

    ax1 = plt.subplot(len(brazil_stations),1,(i+1))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.hist(all_timeseries, bins = n_bins, color = 'green', edgecolor = 'black', alpha=.6, label='all')
    ax1.hist(epidemic_timeseries, bins = n_bins, color = 'blue', edgecolor = 'black', alpha=.6, label='epidemic')
    ax1.hist(non_epidemic_timeseries, bins = n_bins, color = 'red', edgecolor = 'black', alpha=.6, label='non-epidemic')
    plt.grid(linestyle = 'dotted', linewidth = .8)
    ax1.set_ylabel( '{0}'.format(station))
    if i == 0:
        plt.title('Histograms of epidemic and non-epidemic years for 1 variable')
    #plt.ylabel('Mu={0}, std={1} \n Skew={2}  Kurt={3}'.format(mu, std, skew, kurtosis))

    if i > 3:
        ax1.yaxis.set_label_position("right")
plt.subplots_adjust(left=.15, bottom=.05, right=.85, top=.95, wspace=.16, hspace=.50)

'''
fig,ax2 = plt.subplots()
apfs = dict(zip(p_vector, [[] for p in p_vector]))

station = 'Aracaju'
df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
df = df.interpolate() #interpolating missing data
df.index = pd.to_datetime(df.index)
df.sort_index()

for i, p in enumerate(p_vector):
    for t0 in t0_vector:
        apfs[p].append(SVDC_get_apfs(df=df, t0=t0, p=p, years=years, var_name='precip',\
                                               upper_bound=None, verbose=False))
    apfs[p] = np.vstack(apfs[p])
    ax3 = plt.subplot(3, 1, (i+1))
    ax3.hist(apfs[p], bins = 40, color = 'blue', edgecolor = 'black', alpha=.6)
    plt.grid(linestyle = 'dotted', linewidth = .8)
    if i == 0:
        plt.title( 'apf Histogram of {0}'.format(station))
    ax3.set_ylabel('{0}'.format(p))
plt.subplots_adjust(left=.12, bottom=.11, right=.90, top=.88, wspace=.16, hspace=.50)
'''
plt.gcf().set_size_inches([4,20])
plt.savefig('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/EDA/histogram_{0}.png'.format(TARGET), format='png')
