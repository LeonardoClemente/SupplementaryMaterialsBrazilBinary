import pandas as pd
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

### CUSTOM COLORMAP
c1=0
c2=.7
c3=.7
sy = 2006
ey = 2013
v= 4
cdict = {'red':   ((0.0,  c1, c1),
                   (0.70000, .6, .6),
                   (0.70001, 0, 0),
                   (.9500, 0, 0),
                   (.9501, 1, 1),
                   (1.0,  1, 1)),

         'green': ((0.0,  c1, c1),
                   (0.70000, .6, .6),
                   (0.70001, 0, 0),
                   (.9500, 0, 0),
                   (.9501, 1, 1),
                   (1.0,  1, 1)),

         'blue':  ((0.0, c1, c1),
                   (0.70000, .6, .6),
                   (0.70001, 1, 1),
                   (.9500, 1, 1),
                   (.9501, 1, 1),
                   (1.0,  1.0, 1.0))}

custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)

brazil_stations =['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis',\
                  'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos',\
                    'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']




prediction_year = 2017
initial_year = 2008


fig, axarr = plt.subplots(4,5)
axes = axarr.ravel()
for i, st in enumerate(brazil_stations):
    for y in range(initial_year, prediction_year+1):
        data=pickle.load(open('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL/heatmap_dicts/{0}/v6_{1}.p'.format(st, y), 'rb'))
        if  y == initial_year:
            hm = data[0]
        else:
            hm += data[0]
        t_vector = data[1]
        p_vector = data[2]
    t0_start = 0
    t0_end = t_vector.index('2000-09-25')

    p_start = 0
    p_end = len(p_vector)

    index_dates = ['2000-06-01', '2000-07-01', '2000-08-01','2000-09-01']
    index_labels = [d[5:] for d in index_dates]
    index_values = []
    # Get indices for monthly x ticks
    for d in index_dates:
        index_values.append(t_vector.index(d))




    hm /= prediction_year-initial_year+1
    im = axes[i].matshow(hm[p_start: p_end, t0_start:t0_end], cmap=custom_cmap, aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.sca(axes[i])
    plt.title(st)

    if i not in [0,5,10,15]:
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
    else:
        plt.sca(axes[i])
        plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
    if i < 15:
        axes[i].set_xticks([])
        axes[i].set_xticklabels([])
    else:
        plt.sca(axes[i])
        plt.xticks(index_values,index_labels, rotation='90')
        axes[i].xaxis.set_tick_params(labeltop='off', labelbottom='on')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)




plt.show()
