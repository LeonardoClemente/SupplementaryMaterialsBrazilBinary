import pandas as pd
import pickle
from collections import OrderedDict
import time

brazil_stations =['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis',\
                  'Manaus','SaoLuis','BarraMansa', 'Eunapolis', 'Sertaozinho', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos',\
                    'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']

t_list = []
p_list = []
value_list = []
station_list  = []
rename = ['São Gonçalo', 'Santa Cruz', 'Juazeiro do Norte', 'Ji-Paraná', 'Rondonópolis',\
          'Manaus', 'São Luís', 'Barra Mansa', 'Eunápolis', 'Sertãozinho', 'Belo Horizonte',\
          'Parnaíba', 'São Vicente', 'Barretos', 'Aracaju', 'Guarujá', 'Três Lagoas', 'Maranguape',\
          'Barueri', 'Rio de Janeiro']
rename_dict = {}
for i, st in enumerate(brazil_stations):
    rename_dict[st] = rename[i]

prediction_year = 2017
initial_year = 2008

for st in brazil_stations:

    for y in range(initial_year, prediction_year+1):
        data=pickle.load(open('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL/heatmap_dicts/{0}/v6_{1}.p'.format(st, y), 'rb'))
        if  y == initial_year:
            hm = data[0]
        else:
            hm += data[0]
        t_vector = data[1]
        p_vector = data[2]
    hm /= prediction_year-initial_year+1
    for j, t in enumerate(t_vector):
        for i, p in enumerate(p_vector):
            p_list.append(p)
            t_list.append(t)
            value_list.append(hm[i,j])
            station_list.append(rename_dict[st])



df = pd.DataFrame(OrderedDict([('p',p_list), ('t0',t_list), ('value',value_list), ('City',station_list)]))
df['value'][df['value'] == -1] = float('nan')
print(df)

df.to_csv('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL/Sarah_CSV/heatmap_{0}.csv'.format(prediction_year))
