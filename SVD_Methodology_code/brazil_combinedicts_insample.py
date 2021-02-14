import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
dict_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL_insample/heatmap_dicts'

years = list(range(2006,2012))

starting_year = 2006
station = 'Aracaju'
mod='svm'
version=6
experiment_folder = 'BRAZIL_insample'
brazil_stations = ['Manaus', 'Aracaju', 'Barueri', 'Sertaozinho', 'BeloHorizonte']
for station in brazil_stations:
    for y_to_predict in years:

        #get_data
        maps = []
        for j in range(2006, y_to_predict+1):
            data = pickle.load(open("/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/heatmap_dicts/{0}/accumulated_v6_2001_{3}.p".format(station, mod, version, j-1, experiment_folder), 'rb'))
            maps.append(copy.copy(data[0]/len(list(range(2001, j)))))

        decision_map = np.zeros_like(maps[0])

        for map in maps:
            decision_map += map

        decision_map /= len(maps)

        pickle.dump([decision_map], open("/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{4}/heatmap_dicts/{0}/merged_v6_2001_{3}.p".format(station, mod, version, y_to_predict-1, experiment_folder), 'wb'))
