import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator,IndexFormatter
import numpy as np
import copy
import time
from sklearn.utils.extmath import randomized_svd as svd
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from scipy.spatial import distance
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
import pickle
from sklearn import svm
from brazil_functions import *
from brazil_constants import *
from progress.bar import Bar

### CUSTOM COLORMAP
c1=0
c2=.7
c3=.7
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
def SVDC_deploy(df, period_of_interest, variables=['precip', 'temp'], add_runoff_binary=False, prediction_year=2012, epidemic_classification_dict=None, first_training_year=2000, t0_vector=None, p_vector=None, classifier='forest', modes=[0,1], decision_map=None, decision_coordinates=None, decision_values=None, clustering=True, verbose=False, tiebreaker=True):
    '''
    SVD_decision_ensemble performs a decision ensemble based on a series of decision groups generated by
    a clustering analysis.

    #Clustering process. After
    - p_max, p_min: sets the bounds for the period length vector
    - period_of_interest = () #initial and final date that contains the period of interest (poi).
    the period of interest defines the starting and finishing dates for the SVD classifierself.
    e.g. If poi is 01-02-YYYY through 28-02-YYYY, SVD classifier's heatmap will start on 28-02 of previous year and end
    on 01-02 of the next year
    -prediction_year
    -epidemic_classification_dict = dictionary. e.g. {'2001':1, '2002':0, '2003':1}
    '''
    print('Starting SVDC with following parameters:')
    print('Features: {0}'.format(variables))
    print('SVD modes : {0}'.format(modes))
    print('runoff_binary : {0}'.format(add_runoff_binary))
    print('Classifier : {0}'.format(classifier))
    time.sleep(3)
    #Find decision groups
    decision_value_min = decision_values[0]
    decision_value_max = decision_values[1]
    if verbose:
        print('Decision values {0}, {1}'.format(decision_value_min, decision_value_max))
        print('decision_map min and max values: {0}, {1}.'.format(np.min(decision_map), np.max(decision_map)))
        print('MODES:{0}'.format(modes))

    #turns a nxm decision map into a set of samples.
    # matrix = bidimensional numpy array
    if verbose:
        print("Plotting decision map. Please verify everything's correct.")
        a = plt.subplot(1,1,1)
        a_im=a.matshow(decision_map, cmap=plt.cm.hot, aspect='auto', origin='lower')
        plt.ylabel('Decision Map')
        a.yaxis.set_label_position("right")
        a.xaxis.tick_bottom()
        plt.colorbar(a_im, ax=a)
        a.set_xticks(list(range(len(decision_coordinates[0]))), minor=False)
        a.xaxis.set_major_formatter(IndexFormatter(decision_coordinates[0]))
        a.xaxis.set_major_locator(mticker.MaxNLocator(8))
        plt.xticks(rotation=40)
        a.xaxis.set_major_locator(mticker.MaxNLocator(5))
        a.axes.get_xaxis().set_visible(False)
        #plt.show()
        plt.close()


    decision_map[~((decision_map >= decision_value_min) & (decision_map <= decision_value_max))] =0
    rows, columns = np.where((decision_map > 0 ))
    roi = np.vstack([rows,columns]).T


    cluster_weights = []
    cluster_coordinates = []
    cluster_t0 = []
    cluster_p =  []
    total_value_sum = 0



    if clustering:

        #We use a clustering algorithm to find the decision clusters
        db=DBSCAN(eps=5, min_samples=40).fit(roi)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Find coordinates for decision clusters

        indices = [i for i,label in enumerate(labels) if label > -1]
        n_clusters = np.max(labels)+1

        for cluster_number in range(n_clusters):

            cluster_mask = np.equal(labels, cluster_number)
            cluster_indices = [i for i, val in enumerate(cluster_mask) if val==1]

            cluster_sum = 0

            for ind in cluster_indices: # For each sample within cluster

                #get sample coordinates within decision_map
                p_coordinate = roi[ind,0]
                t0_coordinate = roi[ind, 1]

                # get t0 and p coordinates
                cluster_t0.append(decision_coordinates[0][t0_coordinate])
                cluster_p.append(decision_coordinates[1][p_coordinate])
                cluster_sum+=decision_map[p_coordinate,t0_coordinate]

            total_value_sum += cluster_sum
            cluster_coordinates.append(cluster_indices)
            cluster_weights.append(cluster_sum)
    else:
        #If clustering is set to false, we grab all the regions as one big cluster
        cluster_indices = list(range(len(rows)))
        cluster_sum = 0
        n_clusters = 1
        labels = np.array([1]*len(rows))
        if len(cluster_indices)%2 == 0:
            cluster_indices.pop()
        for ind in cluster_indices: # For each sample within cluster
            #get sample coordinates within decision_map
            p_coordinate = roi[ind,0]
            t0_coordinate = roi[ind, 1]

            # get t0 and p coordinates
            cluster_t0.append(decision_coordinates[0][t0_coordinate])
            cluster_p.append(decision_coordinates[1][p_coordinate])
            cluster_sum+=decision_map[p_coordinate,t0_coordinate]

        total_value_sum += cluster_sum
        cluster_coordinates.append(cluster_indices)
        cluster_weights.append(cluster_sum)

    cluster_weights = np.array(cluster_weights)/total_value_sum #Normalizing
    all_indices = np.hstack(cluster_coordinates)

    if verbose:
        print('{0} decision clusters were found'.format(np.max(labels+1)))
        print('Plotting clustered grid, displaying only areas of interest and clusters with different colors')
        clustered_grid = np.zeros_like(decision_map)
        for i,label in enumerate(labels):
            clustered_grid[roi[i,0],roi[i,1]]=label+1
        fig = plt.figure()
        a= plt.subplot(2,1,1)
        a.matshow(decision_map, cmap=plt.cm.hot, aspect='auto', origin='lower')
        #a.colorbar()
        plt.title('Original decision_map')
        b=plt.subplot(2,1,2)
        b.matshow(clustered_grid, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=np.max(labels)+1) # pl is pylab imported a pl plt.cm.hot
        #a.colorbar()
        plt.title('Clusters with classifying acc {0} to {1}'.format(decision_value_min, decision_value_max))
        #plt.show()
        time.sleep(1)
        plt.close()


    if verbose:
        print('Cluster_weights = {0}'.format(cluster_weights))
        print('cluster_coordinates = {0}'.format(cluster_coordinates))
        print('all_indices {0}'.format(all_indices[:]))

    #Generate grid based on p and t0 vectors
    distance_grid = np.zeros_like(decision_map)
    years = []
    for i in range(df.index.shape[0]):
        years.append(df.index[i].year)
    years = sorted(list(set(years)))

    if verbose:
        print(years)

    if prediction_year in years and first_training_year in years:
        training_years = years[years.index(first_training_year): years.index(prediction_year)]
        n_years = len(training_years)
    else:
        print('Missing either prediction_year or first_training_year')
        time.sleep(10)
        return

    '''
    years_before_prediction = years.index(prediction_year)


    training_years = years[0:years_before_prediction]
    n_years = years_before_prediction
    '''


    if verbose:
        print('{0} years detected within dataframe: {1}.'.format(len(years), years))
        print('{0} Years before prediction: {1}'.format(n_years, training_years))

    # check if t0 dates are within poi
    dates_within_poi=[]
    for d in cluster_t0:
        if '{0}'.format(prediction_year) + d[4:] in df[period_of_interest[0]:period_of_interest[1]].index:
            dates_within_poi.append(d)

    if len(dates_within_poi) > 0:
        print('{0} dates from t0_vector are inside period_of_interest range: {1}'.format(len(dates_within_poi),dates_within_poi))


    #Enter main loop
    print('Initiating heatmap loop.')
    bar = Bar('Processing', max=len(cluster_p))
    for p, t0, ind in zip(cluster_p, cluster_t0, all_indices):
        bar.next()
        i = roi[ind,0]
        j = roi[ind,1]

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

            # Perform svd
            U, sigma, VT = svd(X_train, n_components =3, n_iter=15, random_state=None)
            U, sigma, VT = svd(X_train, n_components =3, n_iter=15, random_state=None)
            projections = sigma.reshape([-1,1])*VT
            projections = projections.T

            projections = np.vstack([projections[:,modes], np.matmul(X_predict.reshape([1,-1]),U[:,modes])])

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

            '''
            Now that we got our projections from SVD we can create the classifier
            '''
            if classifier == 'svm':
                mod = svm.SVC(kernel='rbf', gamma=1, C=1, cache_size=400, max_iter=100000)

            elif classifier == 'forest':
                mod = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)

            if verbose:
                print('Fitting with projections shape {0}'.format(projections.shape))
                print(Y_train, training_years)

            mod.fit(classifier_dataset_train, Y_train.ravel())
            pred = mod.predict(classifier_dataset_predict.reshape(1,-1))
            distance_grid[i,j] = pred

    bar.finish()
    cluster_decisions = []
    for cluster_number in range(n_clusters):
        accumulated_decision = 0
        indices = cluster_coordinates[cluster_number]

        for p_coordinate, t0_coordinate in roi[indices,:]:
            accumulated_decision += distance_grid[p_coordinate,t0_coordinate]*decision_map[p_coordinate, t0_coordinate] #Decision weighted by classifier accuracy

        if accumulated_decision > 0:
            cluster_decisions.append(1)
        elif accumulated_decision < 0:
            cluster_decisions.append(-1)
        else:
            cluster_decisions.append(0)


    cluster_decisions = np.array(cluster_decisions)
    final_decision = np.sum(cluster_decisions*cluster_weights)


    if verbose:
        fig, axarr = plt.subplots(3,1,figsize=[4.5,10])

        #axarr[0]= plt.subplot(3,1,1)
        a_im=axarr[0].matshow(decision_map, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=0, vmax=1)
        axarr[0].axes.get_xaxis().set_visible(False)
        plt.colorbar(a_im, ax=axarr[0])
        axarr[0].set_title('Decision map (Year to predict = {2}) \n (classifying acc {0} to {1})'.format(decision_value_min, decision_value_max, prediction_year))
        axarr[0].yaxis.set_label_position("right")
        #b=plt.subplot(3,1,2)
        b_im=axarr[1].matshow(clustered_grid, cmap=plt.cm.tab20c, aspect='auto', origin='lower', vmin=0, vmax=np.max(labels)+1) # pl is pylab imported a pl plt.cm.hot
        plt.colorbar(b_im, ax=axarr[1])
        axarr[1].axes.get_xaxis().set_visible(False)
        axarr[1].yaxis.set_label_position("right")
        axarr[1].set_title('Clusters (N={0})'.format(n_clusters))
        #c=plt.subplot(3,1,3)
        c_im=axarr[2].matshow(distance_grid, cmap=plt.cm.hot, aspect='auto', origin='lower', vmin=-1, vmax=1) # pl is pylab imported a pl plt.cm.hot
        plt.colorbar(c_im, ax=axarr[2])
        axarr[2].set_title('Cluster decisions (final={0})'.format(final_decision))
        axarr[2].yaxis.set_label_position("right")
        axarr[2].set_xticks(list(range(len(decision_coordinates[0]))), minor=False)
        axarr[2].xaxis.set_major_formatter(IndexFormatter(decision_coordinates[0]))
        axarr[2].xaxis.set_major_locator(mticker.MaxNLocator(8))
        axarr[2].xaxis.tick_bottom()
        plt.xticks(rotation=40)
        plt.close()
    else:
        axarr=None
        fig=None

    votes_against=np.sum(distance_grid[distance_grid==-1])
    votes_favor = np.sum(distance_grid[distance_grid==1])
    total_votes = len(cluster_p)

    if verbose:
        print('Decision ensemble finished with the following votes for each cluster (1 in favor, -1 against) \n')
        for c in range(n_clusters):
            print('Cluster {0} = {1}'.format(c+1, cluster_decisions[c]))
        print('Puntual Decision (decision for each classifier) distribution. \n \
              {0} in favor ({1}%).\n {2} against ({3}%). \n Total votes {4}'.format(votes_favor, votes_favor/total_votes, votes_against, votes_against/total_votes,total_votes))

    return final_decision, cluster_decisions, cluster_weights, distance_grid, fig, votes_favor, total_votes


# Files and folders
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/SVDC_DATA_allvars'
experiment_folder = 'BRAZIL_combinedheatmaps'


brazil_stations =['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis',\
                  'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BarraMansa', 'Eunapolis',\
                   'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos',\
                    'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']


#brazil_stations = ['Manaus', 'Aracaju', 'Barueri', 'Sertaozinho', 'BeloHorizonte']

## MODEL SELECTION
mod = 'svm' #svm or forest
version = 6
add_runoff_binary = False
add_peaks = False
decision_map_nyears = 6
n_years_before_prediction = 'all'
variables=['precip', 'temp']
test_years = [2012,2013,2014,2015,2016,2017] # 2012,2013,2014,

if version == 0:
    modes = [0]
elif version == 1:
    modes = [0,1]
elif version == 2:
    modes = [0,1,2]
elif version == 3:
    modes = [0]
    add_peaks = True
elif version == 4:
    modes = [0,1]
    add_peaks = True
elif version == 5:
    modes = [0,1,2]
    add_peaks = True
elif version == 6:
    modes = [1]
    add_peaks = False

tv_df = {}
vf_df = {}
for station in brazil_stations:
    total_votes = []
    votes_favor = []
    results_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/{0}/{1}_{2}.p'.format(station, mod, version, experiment_folder)
    insample_results_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/{0}/{1}_{2}_insample.p'.format(station, mod, version, experiment_folder)

    # Create a vector of t0 and p parameters
    def gendates(year):
        months = list(range(6,13))
        days = list(range(1,32))
        t0_vector = generate_date_vector([year], months, days) + generate_date_vector([year+1], list(range(1,3)), days)

        index_dates = ['{0}-06-01'.format(year), '{0}-07-01'.format(year), '{0}-08-01'.format(year),'{0}-09-01'.format(year), '{0}-10-01'.format(year),\
                       '{0}-11-01'.format(year) ,'{0}-12-01'.format(year), '{0}-01-01'.format(year+1), '{0}-02-01'.format(year+1)]
        non_existent_dates = ['{0}-02-29'.format(year+1), '{0}-02-30'.format(year+1), '{0}-02-31'.format(year+1),\
                              '{0}-06-31'.format(year),'{0}-09-31'.format(year), '{0}-11-31'.format(year)]
        index_values = []
        for d in index_dates:
            index_values.append(t0_vector.index(d))

         # Removing unexistent dates
        for d in non_existent_dates:
            t0_vector.remove(d)
        return t0_vector, index_dates, index_values
    epidemic_classification_per_station = EPIDEMIC_CLASSIFICATION_PER_STATION
    epidemic_classification_dict = dict(zip([y for y in range(2001,2018)],epidemic_classification_per_station[station]))

    normalize = True
    p_vector = list(range(10,100))
    t0_vector, index_dates, index_values = gendates(2001)

    # process
    full_df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
    full_df = full_df.interpolate() #interpolating missing data
    full_df.index = pd.to_datetime(full_df.index)
    # Objective, we want to build a heatmap from SVD samples_labels
    #load decision maps ------



    data = pickle.load(open("/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/heatmap_dicts/{0}/v6_ALL_YEARS_combined.p".format(station, mod, version, experiment_folder), "rb" )) #v{2}_ALL_YEARS.p".format(station, mod, version), "rb" ))


#    data = pickle.load(open(\
#            "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/heatmap_dicts/{0}/accumulated_v6_2001_2005.p".format(station, mod, version, experiment_folder), "rb" )) #v{2}_ALL_YEARS.p".format(station, mod, version), "rb" ))
    decision_map_dict = data[0]
    t0_vector = data[1]
    p_vector = data[2]


    #limit decision map to area of interest
    t0_start = 0
    t0_end = t0_vector.index('2000-10-01')

    p_start = 0
    p_end = len(p_vector)

    decision_values = [.8, 1]  #Only interested in classifiers with 80% acc or above

    decision_score = []
    final_decisions = []
    cluster_decisions_list =[]
    cluster_decision_maps = []
    cluster_weights_list = []
    #test_years = list(range(2006,2012))

    '''
    Fine-Tuning Phaseself.

    In this phase we test our model using the data which we built the model.

    We use a decision map built by stacking the heatmaps for the years 2006-2012
    and use it to predict those same years. If our model is capable of predicting
    all years without trouble we should be good to go and do out of sample testing.



    findings: Model is really sensitive to time windows. Removing one point in the classification
    reduced accuracy considerably. using a static time window made it harder to work with

    '''

    '''
    in_sample_years = list(range(2006,2012))
    decision_map = np.zeros_like(decision_map_dict[2006])
    for y in in_sample_years:
        decision_map+=decision_map_dict[y]
    decision_map /= len(in_sample_years)


    for v in decision_map[p_start: p_end, t0_start: t0_end].ravel():
        if v > .9:
            print(v)

    im=plt.matshow(decision_map[p_start: p_end, t0_start: t0_end], cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)

    plt.colorbar()
    plt.show()

    plt.close()


    for prediction_year in in_sample_years:

        decision_map = np.zeros_like(decision_map_dict[2006])

        if prediction_year == 2006:
            dmy = [2006, 2007, 2008]
        else:
            dmy = list(range(2006, prediction_year))
        for y in dmy:
            decision_map+=decision_map_dict[y]
        decision_map /= len(dmy)

        first_training_year=2000

        t0_vector, index_dates, index_values = gendates(first_training_year)
        decision_coordinates = [t0_vector[t0_start: t0_end], p_vector[p_start: p_end]]
        decision_map = decision_map[p_start: p_end, t0_start: t0_end]
        max_score = np.amax(decision_map[:,index_values[0]:index_values[4]-3])
        print(max_score)
        decision_values = [max_score-.1, max_score]
        #if prediction_year == 2015: decision_values=[max_score-.40, max_score]
        print('Computing heatmap loop for {0} and year {1}.'.format(station, prediction_year))
        final_decision, cluster_decisions, cluster_weights, distance_grid, fig = SVDC_deploy(full_df, period_of_interest=['{0}-03-01'.format(prediction_year), \
                            '{0}-05-31'.format(prediction_year)], prediction_year=prediction_year, \
                            classifier=mod, variables=variables,, modes=modes, verbose=True, first_training_year= first_training_year, \
                            decision_map=decision_map[:,index_values[0]:index_values[4]-3], decision_coordinates=decision_coordinates, \
                            decision_values=decision_values, epidemic_classification_dict=epidemic_classification_dict, clustering=False, add_runoff_binary=add_runoff_binary)
        print('Class:{0}. Prediction:{1}'.format(epidemic_classification_dict[prediction_year], final_decision))


        plt.matshow(distance_grid)
        plt.close()
        if fig is not None:
            insample_fig_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/ensemble_results/{0}/{1}_insample_{2}_{3}.png'.format(station,prediction_year, mod, version, experiment_folder)
            fig.savefig(insample_fig_path, format='png', dpi=300)
        final_decisions.append(final_decision)
        cluster_decisions_list.append(cluster_decisions)
        cluster_decision_maps.append(distance_grid)
        cluster_weights_list.append(cluster_weights)
        print(final_decision, epidemic_classification_dict[prediction_year])
        decision_score.append(epidemic_classification_dict[prediction_year]-final_decision)

    pickle.dump([final_decisions, cluster_decisions_list, cluster_decision_maps, cluster_weights_list, decision_score], open(insample_results_path, 'wb'))
    '''

    '''
     OUT OF SAMPLE PHASE
    '''
    decision_score = []
    final_decisions = []
    cluster_decisions_list =[]
    cluster_decision_maps = []
    cluster_weights_list = []

    for prediction_year in test_years:
        #new_data =  pickle.load(open(\
        #        "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/heatmap_dicts/{0}/accumulated_v{2}_2001_{3}.p".format(station, mod, version, prediction_year-1, experiment_folder), "rb" ))
        #decision_map = new_data[0]/(len(list(range(2001,prediction_year))))
        decision_map = np.zeros_like(decision_map_dict[2006])
        first_training_year=2000

        t0_vector, index_dates, index_values = gendates(first_training_year)
        '''
        for y in range(2006, 2012):
            decision_map+=decision_map_dict[y]
        decision_map /= decision_map_nyears #normalize to range 0-1
        '''

        for y in range(prediction_year-decision_map_nyears, prediction_year):
            decision_map+=decision_map_dict[y]
        decision_map /= decision_map_nyears #normalize to range 0-1


        decision_coordinates = [t0_vector[t0_start: t0_end], p_vector[p_start: p_end]]
        decision_map = decision_map[p_start: p_end, t0_start: t0_end]
        max_score = np.amax(decision_map[:,index_values[0]:index_values[4]-3])

        decision_values = [max_score-accuracy_window[station], max_score]
        #if prediction_year == 2015: decision_values=[max_score-.40, max_score]
        print('Computing heatmap loop for {0} and year {1}.'.format(station, prediction_year))
        final_decision, cluster_decisions, cluster_weights, distance_grid, fig, vf, tv = SVDC_deploy(full_df, period_of_interest=['{0}-03-01'.format(prediction_year), \
                            '{0}-05-31'.format(prediction_year)], variables=variables, prediction_year=prediction_year, \
                            classifier=mod, modes=modes, verbose=True, first_training_year= first_training_year, \
                            decision_map=decision_map, decision_coordinates=decision_coordinates, \
                            decision_values=decision_values, epidemic_classification_dict=epidemic_classification_dict, clustering=False, add_runoff_binary=add_runoff_binary)

        if fig is not None:
            fig_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/ensemble_results/{0}/{1}_{2}_{3}.png'.format(station,prediction_year, mod, version, experiment_folder)
            fig.savefig(fig_path, format='png', dpi=300)
        final_decisions.append(final_decision)
        cluster_decisions_list.append(cluster_decisions)
        cluster_decision_maps.append(distance_grid)
        cluster_weights_list.append(cluster_weights)
        decision_score.append(epidemic_classification_dict[prediction_year]-final_decision)
        votes_favor.append(vf)
        total_votes.append(tv)

    vf_df[station] = np.array(votes_favor)
    tv_df[station] = np.array(total_votes)
    pickle.dump([final_decisions, cluster_decisions_list, cluster_decision_maps, cluster_weights_list, decision_score], open(results_path, 'wb'))


vf_df =pd.DataFrame(vf_df, index=test_years)
tv_df = pd.DataFrame(tv_df, index=test_years)

tv_df.to_csv('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/total_votes.csv'.format(station, mod, version, experiment_folder))
vf_df.to_csv('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/votes_favor.csv'.format(station, mod, version, experiment_folder))
