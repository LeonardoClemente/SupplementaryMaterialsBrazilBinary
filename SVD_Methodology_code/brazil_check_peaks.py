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
import seaborn as sns


# Files and folders
heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/model_compare/heatmap_dicts'
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'

brazil_stations =['Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2']
station = 'Aracaju_MERRA_2'
# process
full_df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
full_df = full_df.interpolate() #interpolating missing data
full_df.index = pd.to_datetime(full_df.index)
# Objective, we want to build a heatmap from SVD samples_labels

# Objective, we want to build a heatmap from SVD samples_labels

# Create a vector of t0 and p parameters
normalize = True
p_vector = list(range(10,100))
years = [2000]
months = list(range(6,13))
days = list(range(1,32))

t0_vector = generate_date_vector(years, months, days) + generate_date_vector([2001], list(range(1,3)), days)

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

ystart=2013
yend=2018


p_vector = [25, 50, 90]
t0 = '2000-06-01'
years = list(range(2001,2004))

# Test del mismo date diferentes p
apf_dict = {}
for p in p_vector:

        apfs, list_apfs = SVDC_get_apfs(df=full_df, t0=t0, p=p, years=years, upper_bound=None, normalize=True, verbose=False)
        print(apfs)
        apf_dict[p] = list(apfs)


array_apfs=np.vstack(list_apfs)
print('ARREGLO DE TIMESERIES')
print(array_apfs)
plt.plot(array_apfs.transpose())
plt.legend()
plt.show()
plt.close()

print(years)
apf_df = pd.DataFrame(apf_dict, index=years)
print(apf_df)
time.sleep(10)
im = sns.heatmap(apf_df.transpose())
plt.show()




s='0.91024916  1.82075494  2.39942826  1.71555353  1.44603207  3.39335851 1.36003352  3.89467982  5.43098244  3.34685335  2.43080773  0.89615311 1.3631424  1.54134376  2.38808727  5.61359219  1.59122011  1.98091232 1.76486486  1.22720322  1.94373478  3.35359312  3.46614234  2.22208359  1.10092862'.replace('  ', ',')
