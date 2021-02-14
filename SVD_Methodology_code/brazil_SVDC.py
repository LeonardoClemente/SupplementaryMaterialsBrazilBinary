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
from brazil_functions import *
from progress.bar import Bar
import collections




def SVD_classifier(df, period_of_interest, prediction_year=2012, \
                   epidemic_classification_dict=None, training_year_window='ALL',\
                    t0_vector=None, p_vector=None, classifier='SVM', modes=[0,1], verbose=False):
    '''
    - p_max, p_min: sets the bounds for the period length vector
    - period_of_interest = () #initial and final date that contains the period of interest (poi).
    the period of interest defines the starting and finishing dates for the SVD classifierself.
    e.g. If poi is 01-02-YYYY through 28-02-YYYY, SVD classifier's heatmap will start on 28-02 of previous year and end
    on 01-02 of the next year
    -prediction_year
    -epidemic_classification_dict = dictionary. e.g. {'2001':1, '2002':0, '2003':1}
    '''


    #Generate grid based on p and t0 vectors
    distance_grid = np.zeros([len(p_vector),len(t0_vector)])


    years = []
    for i in range(df.index.shape[0]):
        years.append(df.index[i].year)
    years = sorted(list(set(years)))

    years_before_prediction = years.index(prediction_year)

    if training_year_window == 'ALL':
        training_years = years[0:years_before_prediction]
        n_years = years_before_prediction
    elif training_year_window < years_before_prediction:
        training_years = years[years_before_prediction-training_year_window:years_before_prediction]
        n_years = training_year_window
    else:
        print("Can't retrieve training window: {0}. Place make sure training window is 'ALL' or an int number within the number of years size".format(training_year_window))

    if verbose:
        print('{0} years detected within dataframe: {1}.'.format(len(years), years))
        print('{0} Years before prediction: {1}'.format(n_years, training_years))


    # check if t0 dates are within
    dates_within_poi=[]
    for d in t0_vector:
        if '{0}'.format(prediction_year) + d[4:] in df[period_of_interest[0]:period_of_interest[1]].index:
            dates_within_poi.append(d)

    if len(d) > 0:
        print('{0} dates from t0_vector are inside period_of_interest range: {1}'.format(len(dates_within_poi),dates_within_poi))


    #Enter main loop
    print('Initiating heatmap loop.')
    bar = Bar('Processing', max=len(p_vector))
    for i, p in enumerate(p_vector):
        bar.next()
        for j, t0 in enumerate(t0_vector):


            if verbose: print('Reshaping data')
            X = SVDC_reshape_yearly_data_stolerman(df=df, t0=t0, p=p,\
                                                   years=training_years, \
                                                   upper_bound=period_of_interest[0],\
                                                   normalize=True, verbose=False)

            if verbose: print('Reshaping data done')

            '''
            Each column of X represents one year of data in the order of years_before_prediction. If we want out classification at year Y
            we need Y-1 as out of sample input and Y-2, Y-3...1 as our training dataset. As we're trying to classify every Y with previous year data, we also assign
            the epidemic classification of year Y to the label for Y-1
            '''
            if X is not None:

                X_train = X[:,:-1]
                X_predict = X[:,-1]
                Y_train = []
                for year in training_years[:-1]: # Can take out of loop but keeping for clear reading purposes
                    Y_train.append(epidemic_classification_dict[year+1])

                Y_train=np.vstack(Y_train)
                Y_predict = epidemic_classification_dict[prediction_year]

                # Perform svd
                U, sigma, VT = svd(X_train, n_components =3, n_iter=15, random_state=None)
                projections = sigma.reshape([-1,1])*VT
                projections = projections.T
                projections = projections[:,modes]


                '''
                Now that we got our projections from SVD we can create the classifier
                '''
                mod = svm.SVC(kernel='rbf', gamma=1, C=1, cache_size=400, max_iter=100000)
                if verbose: ('Fitting with projections shape {0} and target shape {1}'.format(projections.shape, Y_predict))
                mod.fit(projections, Y_train.ravel())
                pred = mod.predict(np.matmul(X_predict.reshape([1,-1]),U[:,modes]))

                distance_grid[i,j] = (pred == Y_predict)
            else:
                distance_grid[i,j] = -1
    bar.finish()
    return distance_grid



# Files and folders
heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS'
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'
brazil_stations =['BeloHorizonte_MERRA_2', 'Manaus_MERRA_2'] #'Aracaju_MERRA_2',, 'RECIFE (CURADO)', 'ARACAJU', 'BELO HORIZONTE', 'MANAUS',
#station = 'Manaus_MERRA_2'


# Create a vector of t0 and p parameters
normalize = True
p_vector = list(range(10,100))
years = [2001]
months = list(range(6,13))
days = list(range(1,32))

t0_vector = generate_date_vector(years, months, days) + generate_date_vector([2002], list(range(1,3)), days)

index_dates = ['2001-06-01', '2001-07-01', '2001-08-01','2001-09-01', '2001-10-01',\
               '2001-11-01' ,'2001-12-01', '2002-01-01', '2002-02-01']
non_existent_dates = ['2002-02-29', '2002-02-30', '2002-02-31', '2001-06-31','2001-09-31', '2001-11-31',]

index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))

 # Removing unexistent dates
for d in non_existent_dates:
    t0_vector.remove(d)


epidemic_classification_per_station = {
    'ARACAJU': [1,1,0,0,0,1,1,1,0,1,1],
    'MANAUS':[1,1,0,0,0,1,1,0,1,1,1],
    'BELO HORIZONTE':[1,0,0,0,0,1,1,1,1,0,0],
    'RECIFE (CURADO)':[1,0,0,0,0,0,1,0,1,1,1],
    'Aracaju_MERRA_2': [1,1,0,0,0,1,1,1,0,1,1],
    'BeloHorizonte_MERRA_2': [1,0,0,0,0,1,1,1,1,0,0],
     'Manaus_MERRA_2':[1,1,0,0,0,1,1,0,1,1,1]
}


class_color = {0:'r', 1:'b'}
n_years = 11
modes = [0,1] #starts from zero


for station in brazil_stations:

    distance_grids = []
    epidemic_classification_dict = dict(zip([y for y in range(2002,2013)],epidemic_classification_per_station[station]))
    full_df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
    full_df = full_df.interpolate() #interpolating missing data
    full_df.index = pd.to_datetime(full_df.index)
    # Objective, we want to build a heatmap from SVD samples_labels
    full_df.sort_index()
    full_df = full_df['2001-01-01':'2016-12-31']

    for y in range(2007,2013):
        print('Computing heatmap loop for {0} and year {1}.'.format(station, y))
        distance_grid = SVD_classifier(full_df, period_of_interest=['{0}-03-01'.format(y), '{0}-05-31'.format(y)], prediction_year=y, \
                           epidemic_classification_dict=epidemic_classification_dict, training_year_window='ALL',\
                            t0_vector=t0_vector, p_vector=p_vector, classifier='SVM', modes=modes,  verbose=False)
        distance_grids.append(copy.copy(distance_grid))



    distance_grid = np.zeros_like(distance_grid)

    for dg in distance_grids:
        distance_grid += dg
        #im = plt.matshow(dg, cmap=plt.cm.hot, aspect='auto', origin='lower') # pl is pylab imported a pl
        #plt.colorbar()
        #plt.show()

    pickle.dump([distance_grids, t0_vector, p_vector], open("{0}/exp1/SVDC_{1}_all_grids.p".format(heatmap_path, station), "wb" ))
    pickle.dump([distance_grid, t0_vector, p_vector], open("{0}/exp1/SVDC_{1}.p".format(heatmap_path, station), "wb" ))
    '''
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
    '''
    #Choose highest modes
