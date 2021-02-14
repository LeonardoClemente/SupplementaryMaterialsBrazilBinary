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



def SVDC_heatmap_generatorv1(df, period_of_interest, prediction_year=2012, epidemic_classification_dict=None, training_year_window='ALL', t0_vector=None, p_vector=None, classifier='SVM', modes=[0], verbose=False):
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


def SVDC_heatmap_generator(df, period_of_interest, prediction_year=2012, \
                           epidemic_classification_dict=None, training_year_window='ALL', t0_vector=None, \
                           p_vector=None, classifier='SVM', modes=[0], add_peaks=False,\
                           add_runoff_binary=False, verbose=False, variables=['precip', 'temp']):

    '''

    - p_max, p_min: sets the bounds for the period length vector
    - period_of_interest = () #initial and final date that contains the period of interest (poi).
    the period of interest defines the starting and finishing dates for the SVD classifierself.
    e.g. If poi is 01-02-YYYY through 28-02-YYYY, SVD classifier's heatmap will start on 28-02 of previous year and end
    on 01-02 of the next year
    -prediction_year
    -epidemic_classification_dict = dictionary. e.g. {'2001':1, '2002':0, '2003':1}


    v2.:
    Version two of heatmap generators utilized 3 modes rather than 2 and also incorporates the average number of peaks
    as extra dimensions prior to the classifier phase
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
            X = SVDC_reshape_yearly_data_stolerman(df=df[variables], t0=t0, p=p,\
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
                projections = np.vstack([projections[:,modes], np.matmul(X_predict.reshape([1,-1]),U[:,modes])])

                '''
                if not np.equal(projections[-1,:], np.matmul(X_predict.reshape([1,-1]),U[:,modes]).reshape(1,-1)).all():
                    print('WARNING! projections and prediction sample matmul are not equal')
                    print(projections[-1,:], np.matmul(X_predict.reshape([1,-1]),U[:,modes]))
                    time.sleep(10)
                if verbose:
                    print('Verifying predict_projection is correct = {0},{1}, {2}'.format(projections,projection_predict, np.matmul(X_predict.reshape([1,-1]),U[:,modes])))
                '''

                '''
                Merging SVD projections average_peak_frequencies for each year. They should have the same length
                '''

                if add_peaks:
                    # This function returns the delta value stated in Stolerman's paper
                    average_peak_frequencies = SVDC_get_apfs(df=df, t0=t0, p=p,\
                                                           years=training_years, \
                                                           upper_bound=period_of_interest[0],\
                                                           normalize=True, verbose=False)
                    classifier_dataset = np.hstack([projections, average_peak_frequencies])
                else:
                    classifier_dataset = projections

                if add_runoff_binary:
                    # This function returns the delta value stated in Stolerman's paper
                    average_runoff = SVDC_get_runoffbinary(df=df, t0=t0, p=p,\
                                                           years=training_years, \
                                                           upper_bound=period_of_interest[0],\
                                                           normalize=True, verbose=False)
                    classifier_dataset = np.hstack([projections, average_runoff])

                else:
                    classifier_dataset = projections

                classifier_dataset_train = classifier_dataset[:-1,:]
                classifier_dataset_predict = classifier_dataset[-1,:]

                if verbose:
                    print(classifier_dataset_train, classifier_dataset_predict)
                if classifier == 'svm':
                    mod = svm.SVC(kernel='rbf', gamma=1, C=1, cache_size=400, max_iter=100000)
                elif classifier == 'forest':
                    mod = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
                if verbose:
                    ('Fitting with projections shape {0} and target shape {1}'.format(classifier_dataset_train.shape, Y_predict))

                mod.fit(classifier_dataset_train, Y_train.ravel())
                pred = mod.predict(classifier_dataset_predict.reshape(1, -1))
                distance_grid[i,j] = (pred == Y_predict)
            else:
                distance_grid[i,j] = -1
    bar.finish()
    return distance_grid


heatmap_version = 6
modes = [1]
variables=['precip', 'temp', 'humidity']
add_peaks=False
add_runoff_binary=False
# Files and folders
heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL_humidity/heatmap_dicts'
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/SVDC_DATA_allvars'

brazil_stations = ['Manaus', 'Aracaju', 'Barueri', 'Sertaozinho', 'BeloHorizonte']

'''
[ 'Aracaju', 'SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis',\
                   'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BarraMansa',\
                    'Eunapolis', 'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos',\
                     'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio' ]'''


# SAO LUIS 2017 'SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis', 'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BarraMansa', 'Eunapolis', 'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos', 'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio',
#brazil_stations = ['Sertaozinho']
#station = 'Rio'
#full_df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
#full_df = full_df.interpolate() #interpolating missing data
#full_df.index = pd.to_datetime(full_df.index)
# Objective, we want to build a heatmap from SVD samples_labels
#full_df.sort_index()
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

ystart=2006
yend=2018

for station in brazil_stations:

    full_df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
    full_df = full_df.interpolate() #interpolating missing data
    full_df.index = pd.to_datetime(full_df.index)
    # Objective, we want to build a heatmap from SVD samples_labels
    full_df.sort_index()

    epidemic_classification_per_station = EPIDEMIC_CLASSIFICATION_PER_STATION
    epidemic_classification_dict = dict(zip([y for y in range(2001,2018)],epidemic_classification_per_station[station]))

    for y in range(ystart,yend):
        print('Computing heatmap loop for {0} and year {1}.'.format(station, y))
        print('modes used: {0} average days between peaks added: {1}'.format(modes, add_peaks))
        time.sleep(3)
        heatmap = SVDC_heatmap_generator(full_df, period_of_interest=['{0}-01-01'.format(y), '{0}-05-31'.format(y)], prediction_year=y, \
                           epidemic_classification_dict=epidemic_classification_dict, training_year_window='ALL',\
                            t0_vector=t0_vector, p_vector=p_vector, classifier='svm', modes=[1],\
                            variables=variables, verbose=False, add_peaks=add_peaks, add_runoff_binary=add_runoff_binary)
        data = pickle.dump([heatmap, t0_vector, p_vector], open("{0}/{1}/v{2}_{3}.p".format(heatmap_path, station, heatmap_version, y), "wb" ))
