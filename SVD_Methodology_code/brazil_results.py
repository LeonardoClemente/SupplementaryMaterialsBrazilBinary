import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import collections


'''
The results
'''
mod = 'svm'
version = 6
all_stations = ['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis', 'Manaus', 'SaoLuis', 'BarraMansa', 'Eunapolis', 'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos', 'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']
ready =  ['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis', 'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BarraMansa', 'Eunapolis', 'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos', 'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']
#ready=['Manaus', 'Aracaju', 'Barueri', 'Sertaozinho', 'BeloHorizonte'] #
brazil_stations = []

for st in all_stations:
    if st in ready:
        brazil_stations.append(st)

station = 'Sertaozinho'
folder = 'BRAZIL_combinedheatmaps'
accuracy = collections.OrderedDict()
years = list(range(2012,2018))
df = collections.OrderedDict()
for i, station in enumerate(brazil_stations):
    data = pickle.load(open('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/{3}/ensemble_results/{0}/{1}_{2}.p'.format(station, mod, version, folder), 'rb'))
    final_decision = data[0]
    cluster_decisions_list = data[1]
    cluster_decision_maps = data[2]
    cluster_weights_list = data[3]
    print(final_decision)
    decision_score = np.round(data[4])
    df[station] = decision_score
    print(station, decision_score)
    accuracy[station] = np.sum(np.equal(decision_score, 0))/len(decision_score)


df = pd.DataFrame(df, index=years)
ax = sns.heatmap(df.transpose(), linewidths=.5, vmax=2, cmap='coolwarm')
plt.subplots_adjust(left=0.29, bottom=0.11, right=1.0, top=.88, wspace=.20, hspace=.20)
plt.title('{0}'.format(folder))
plt.figure()
ax2 = sns.barplot(data=pd.DataFrame(accuracy, index=['Accuracy']))

plt.show()
