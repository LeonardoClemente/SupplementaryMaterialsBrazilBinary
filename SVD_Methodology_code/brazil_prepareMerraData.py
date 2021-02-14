import pandas as pd
import numpy as np
import time
MERRA_FOLDER = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/MERRA2_allvars'
OUTPUT_FOLDER = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/SVDC_DATA_allvars'
cities = ['Aracaju', 'Sertaozinho', 'Rio', 'JuazeirodoNorte', 'SaoVicente', 'Parnaiba', 'Barretos', 'SaoGoncalo', 'Manaus', 'JiParana', 'Eunapolis', 'TresLagoas', 'SaoLuis', 'Guaruja', 'SantaCruz', 'Barueri', 'Rondonopolis', 'Maranguape', 'BarraMansa', 'BeloHorizonte']

dates = pd.date_range(start='2000-01-01', end='2017-12-31').strftime("%Y-%m-%d")
v = np.zeros(len(dates))
sample_df = pd.DataFrame({'zeros':v}, index=dates)

print(sample_df)

for i, city in enumerate(cities):

  if i == 0:
      df=pd.read_csv('{0}/{1}_MERRA_2_allvars.csv'.format(MERRA_FOLDER,city), index_col=[0])
      ind = df.index

  else:
      df= pd.read_csv('{0}/{1}_MERRA_2_allvars.csv'.format(MERRA_FOLDER,city))
      df.set_index(ind, inplace=True)

  names=list(df)
  new_df = pd.concat([sample_df, df[['precip', 'temp', 'humidity', 'runoff_binary', 'runoff_diff']]], axis=1)
  print(new_df)
  new_df.to_csv('{0}/{1}.csv'.format(OUTPUT_FOLDER,city))
