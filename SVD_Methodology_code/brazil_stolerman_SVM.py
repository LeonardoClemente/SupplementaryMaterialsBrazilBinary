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



# Get periods

def generate_date_vector(years, months, days):

    date_vector = []
    for i in years:
        for j in months:
            for k in days:
                if k < 10 and j < 10:
                    date_vector.append('{0}-0{1}-0{2}'.format(i, j, k))
                elif k >= 10 and j < 10:
                    date_vector.append('{0}-0{1}-{2}'.format(i, j, k))
                elif k < 10 and j >= 10:
                    date_vector.append('{0}-{1}-0{2}'.format(i, j, k))
                elif k >= 10 and j >= 10:
                    date_vector.append('{0}-{1}-{2}'.format(i, j, k))

    return date_vector



def average_peak_frequency(vector, verbose=False):
    peak_indices = detect_peaks(vector, mph=0, mpd=1, threshold=0, edge='rising',
                     kpsh=False, valley=False, show=False, ax=None)
    if verbose:
        print(peak_indices)

    if len(peak_indices)>0:
        delta = 1/np.mean(np.hstack([peak_indices[0],np.diff(peak_indices)])) # Average distance (days) between precipitation peaks
    else:
        delta = 0
    return delta

def map_timeseries_to_points(df=None, t0='2002-03-01', p=40, n_years=11,\
                             operation_dict=None, operation_per_variable=None, \
                             year_epidemic_classification=None, normalize=False, verbose=False):

    # Converts meteorological timeseries based on a specific operations for each variable
    y = int(t0[0:4])
    samples = []
    for i in range(0,n_years):
        values = []
        t = '{0}'.format(y+i) + t0[4:]

        ind = df.index.get_loc(t)
        timeseries = df[ind:ind+p]

        if verbose:
            print(timeseries)

        if normalize:
            timeseries = standardize_df(timeseries)

        if verbose:
            print(timeseries)

        for variable, operation in operation_per_variable.items():
            values.append(operation_dict[operation](timeseries[variable].values))

        values.append(year_epidemic_classification[i])
        samples.append(values)
    X = np.vstack(samples)

    return X


def calculate_subgrid_ranges(p_vector=None, t0_vector=None, subgrid_size=(6,5)):
    # subgrid_size = (t0_subsize, p_subsize)
    t0_divisions = len(t0_vector)//subgrid_size[1]
    p_divisions = len(p_vector)//subgrid_size[0]
    ranges = []


    for i in range(t0_divisions):
        for j in range(p_divisions):
            ranges.append( [ (i*subgrid_size[1], (i+1)*subgrid_size[1]) , (j*subgrid_size[0], (j+1)*subgrid_size[0])  ] )

    return ranges

def generate_samples(df=None, p_vector=None, t0_vector=None, subgrid_range=None, \
                     year_epidemic_classification=None, operation_dict=None,\
                     operation_per_variable=None):
    sub_datasets = []
    for i in range(subgrid_range[0][0], subgrid_range[0][1]):
        for j in range(subgrid_range[1][0], subgrid_range[1][1]):

            sub_dataset = map_timeseries_to_points(df=df, t0=t0_vector[i], p=p_vector[j], n_years=11,\
                                             operation_dict=operation_dict, operation_per_variable=operation_per_variable, \
                                             year_epidemic_classification=year_epidemic_classification)
            sub_datasets.append(sub_dataset)

    X = np.vstack(sub_datasets)
    return X

def plot_region(labeled_samples=None, class_color=None, verbose=True):

    for i in range(np.size(labeled_samples, axis=0)):
        print()
        plt.scatter(labeled_samples[i,0],labeled_samples[i,1], c=class_color[labeled_samples[i,2]])

    plt.show()


def standardize_df(df=None):
    #Normalize each column of a dataframe
    df_names = list(df)

    for name in df_names:
        mu = df[name].values.mean()
        sigma = df[name].values.std()
        df[name] = (df[name]-mu)/sigma
    return df

def standardize_matrix(matrix=None):

    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    matrix -= mu
    matrix /= sigma

    return matrix

def make_meshgrid(x, y, h=.02, space=.2):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_d = x.max()-x.min()
    y_d = y.max()-y.min()
    x_min, x_max = x.min() - x_d*space, x.max() + x_d*space
    y_min, y_max = y.min() - y_d*space, y.max() + y_d*space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def svm_test(labeled_samples=None, class_color=None, C=1):

    X0, X1, labels = labeled_samples[:, 0], labeled_samples[:, 1], labeled_samples[:,2]
    x, y = make_meshgrid(X0, X1)
    models = (svm.SVC(kernel='linear', C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C))
    models = (m.fit(labeled_samples[:,:2], labeled_samples[:,2]) for m in models)

    titles = ('SVC with linear kernel',
              'SVC with RBF kernel')

    fig, sub = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, x, y,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel('Peak Frequency')
        ax.set_ylabel('Average Temperatue')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()
    plt.close()

def test_accuracy(mod,validation_dataset):
    X_pred = validation_dataset[:,:2]
    labels = validation_dataset[:,2]
    z = mod.predict(X_pred)
    acc = np.sum(z==labels)/len(z)
    return acc

def separate_dataset(labeled_samples, training=.8):
    n_samples = np.size(labeled_samples,axis=0)
    n_training = int(n_samples*.8)

    randomized_indices = np.random.permutation(list(range(n_samples)))
    training_dataset = labeled_samples[randomized_indices[0:n_training],:]
    validation_dataset = labeled_samples[randomized_indices[n_training:],:]

    return training_dataset, validation_dataset

def loop_svm(labeled_samples=None, n_times=100, kernel='rbf'):
    acc = []

    for i in range(n_times):
        mod = svm.SVC(kernel=kernel, gamma=1, C=1, cache_size=400, max_iter=100000)
        training_dataset, validation_dataset = separate_dataset(labeled_samples)


        mu = np.mean(validation_dataset[:,:2], axis=0)
        sigma = np.std(training_dataset[:,:2], axis=0)
        training_dataset[:,:2] = (training_dataset[:,:2]-mu)/sigma
        validation_dataset[:,:2] = (validation_dataset[:,:2] - mu)/sigma

        mod.fit(training_dataset[:,:2], training_dataset[:,2])
        acc.append(test_accuracy(mod, validation_dataset))

    acc = np.array(acc).sum()/len(acc)

    return acc




operation_dict = {
    'average':np.mean,
    'average_peak_frequency':average_peak_frequency
}

operation_per_variable = {
    'Precipitacao':'average_peak_frequency',
    'Temp Comp Media':'average'
}


#START

# Files and folders
heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS'
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'

# Read in data
full_df = pd.read_csv('{0}/ACARAJU.csv'.format(csv_data_path), skiprows=16)


#del full_df['Data']
#full_df.set_index(full_df['Data'], inplace=True)


interpolate_df = full_df.interpolate()

rain  = full_df[full_df['Hora'] == 0]['Precipitacao'].to_frame()
temp  = full_df[full_df['Hora'] == 0]['Temp Comp Media'].to_frame()

full_df = pd.concat([rain, temp], axis=1).interpolate() #interpolating missing data

all_dates = pd.date_range(start='2001-01-01', end='2012-12-31').strftime("%Y-%m-%d")

full_df.set_index(all_dates, inplace=True)

#full_df.set_index(pd.date_range(start='2001-01-01', end='2012-12-31'), inplace=True)

# Objective, we want to build a heatmap from SVD samples_labels

# Create a vector of t0 and p parameters
normalize = False
p_vector = list(range(10,100))

years = [2001]
months = list(range(6,13))
days = list(range(1,32))

t0_vector = generate_date_vector(years, months, days) + generate_date_vector([2002], list(range(1,3)), days)

index_dates = ['2001-06-01', '2001-07-01', '2001-08-01','2001-09-01', '2001-10-01',\
               '2001-11-01' ,'2001-12-01', '2002-01-01', '2002-02-01']
non_existent_dates = ['2002-02-29', '2002-02-30', '2002-02-31', '2001-06-31','2001-09-31', '2001-11-31',]

index_values = []


kernel = 'linear'
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))



 # Removing unexistent dates

for d in non_existent_dates:
    print(d)
    t0_vector.remove(d)



p_subrange = 6
t0_subrange = 5
distance_grid = np.zeros([ (len(p_vector)//p_subrange)*p_subrange, (len(t0_vector)//t0_subrange)*t0_subrange])

# year_epidemic_classification (epidemic = 0, non-epidemic = 1)

year_epidemic_classification = [0,0,1,1,1,0,0,0,1,0,0]
class_color = {0:'r', 1:'b'}
n_years = 11
year_epidemic_classification = year_epidemic_classification[0:n_years]
modes = [0,1] #starts from zero

subgrid_ranges = calculate_subgrid_ranges(p_vector=p_vector, t0_vector=t0_vector, subgrid_size=(6,5))
#Enter main loop
for subgrid_range in subgrid_ranges:

    labeled_samples = generate_samples(full_df, p_vector, t0_vector, \
                                       subgrid_range, year_epidemic_classification,\
                                        operation_dict, operation_per_variable)

    #plot_region(labeled_samples, class_color=class_color)
    #svm_test(labeled_samples, class_color=class_color)

    accuracy = loop_svm(labeled_samples=labeled_samples, n_times=100, kernel=kernel)
    print(accuracy, subgrid_range)
    distance_grid[subgrid_range[1][0]:subgrid_range[1][1], subgrid_range[0][0]:subgrid_range[0][1]] = accuracy




data = pickle.dump([distance_grid, t0_vector, p_vector], open("{0}/SVM_{1}_norm.p".format(heatmap_path, kernel), "wb" ))
im = plt.matshow(distance_grid, cmap=plt.cm.hot, aspect='auto', origin='lower') # pl is pylab imported a pl

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
