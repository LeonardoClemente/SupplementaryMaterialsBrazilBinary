import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

brazil_stations =['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis',\
                  'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BarraMansa', 'Eunapolis',\
                   'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos',\
                    'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']


for station in brazil_stations:
 
    heatmap = {}
    for y in range(2006,2017):

        s = pickle.load(open('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL_heatmap_dicts/{1}/v6_{0}.p'.format(y, station),'rb'))
        t = pickle.load(open('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL_insample/heatmap_dicts/{1}/accumulated_v6_2001_{0}.p'.format(y-1, station),'rb'))
        plt.imshow(s[0])
        plt.figure()
        plt.imshow(t[0])
        plt.show()

        m = np.multiply(t[0], s[0])
        m /= np.max(m)

        heatmap[y] =  copy.copy(m)
        print(station)
    s[0] = heatmap
    pickle.dump(s, open('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL_heatmap_dicts/{0}/v6_ALL_YEARS_combined.p'.format(station), 'wb') )
