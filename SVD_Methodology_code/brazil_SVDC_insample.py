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


# Files and folders
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/SVDC_DATA_allvars'
experiment_folder = 'BRAZIL_insample'

'''
brazil_stations =['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis',\
                  'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BarraMansa', 'Eunapolis',\
                   'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos',\
                    'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']
'''

brazil_stations = ['Manaus', 'Aracaju', 'Barueri', 'Sertaozinho', 'BeloHorizonte']

## MODEL SELECTION
mod = 'svm' #svm or forest
modes = [1]
decision_map_nyears = 6
version = 6
add_runoff_binary = False
add_peaks = False
n_years_before_prediction = 'all'
variables=['precip', 'temp']
dict_type = 'merged'

for station in brazil_stations:

    results_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/{0}/{1}_{2}_{3}.p'.format(station, mod, version, experiment_folder, dict_type)
    insample_results_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/{0}/{1}_{2}_{3}_insample.p'.format(station, mod, version, experiment_folder, dict_type)

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


    data = pickle.load(open(\
            "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/heatmap_dicts/{0}/accumulated_v6_2001_2005.p".format(station, mod, version, experiment_folder), "rb" )) #v{2}_ALL_YEARS.p".format(station, mod, version), "rb" ))
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
    #test_years = [2012,2013,2014,2015,2016,2017] # 2012,2013,2014,
    test_years = list(range(2006,2012))

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
        new_data =  pickle.load(open(\
                "/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/heatmap_dicts/{0}/{5}_v{2}_2001_{3}.p".format(station, mod, version, prediction_year-1, experiment_folder, dict_type), "rb" ))

        if dict_type == 'accumulated':
            decision_map = new_data[0]/(len(list(range(2001,prediction_year))))
        elif dict_type == 'merged':
            decision_map = new_data[0]

        '''
        plt.imshow(decision_map)
        plt.colorbar()
        plt.show()
        time.sleep(10)
        '''
        first_training_year=2000

        t0_vector, index_dates, index_values = gendates(first_training_year)

        decision_coordinates = [t0_vector[t0_start: t0_end], p_vector[p_start: p_end]]
        decision_map = decision_map[p_start: p_end, t0_start: t0_end]

        max_score = np.amax(decision_map[:,index_values[0]:index_values[4]-3])

        decision_values = [max_score-.05, max_score]
        #if prediction_year == 2015: decision_values=[max_score-.40, max_score]
        print('Predicting for {0} in year {1}.'.format(station, prediction_year))
        final_decision, cluster_decisions, cluster_weights, distance_grid, fig = SVDC_deploy(full_df, period_of_interest=['{0}-03-01'.format(prediction_year), \
                            '{0}-05-31'.format(prediction_year)], variables=variables, prediction_year=prediction_year, \
                            classifier=mod, modes=modes, verbose=True, first_training_year= first_training_year, \
                            decision_map=decision_map, decision_coordinates=decision_coordinates, \
                            decision_values=decision_values, epidemic_classification_dict=epidemic_classification_dict, clustering=False, add_runoff_binary=add_runoff_binary)

        if fig is not None:
            fig_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/ensemble_results/{0}/{1}_{2}_{3}_{4}.png'.format(station,prediction_year, mod, version, experiment_folder, dict_type)
            fig.savefig(fig_path, format='png', dpi=300)
        final_decisions.append(final_decision)
        cluster_decisions_list.append(cluster_decisions)
        cluster_decision_maps.append(distance_grid)
        cluster_weights_list.append(cluster_weights)
        decision_score.append(epidemic_classification_dict[prediction_year]-final_decision)

    pickle.dump([final_decisions, cluster_decisions_list, cluster_decision_maps, cluster_weights_list, decision_score], open(results_path, 'wb'))
