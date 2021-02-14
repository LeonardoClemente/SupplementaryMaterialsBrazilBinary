import pandas as pd
import numpy as np
import time
main_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'
file_names = ['Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2']

dates = pd.date_range(start='2017-01-01', end='2017-12-31').strftime("%Y-%m-%d")
n_zeros=len(dates)
print(n_zeros)

for file_name in file_names:
  df=pd.read_csv('{0}/{1}.csv'.format(main_folder,file_name), index_col=[0])
  names=list(df)
  z_list= [np.zeros(n_zeros).fill(np.nan) for i in range(len(names))]
  d = dict(zip(names,z_list))
  added_df = pd.DataFrame(d, index=dates)
  df=df.append(added_df)
  df.to_csv('{0}/{1}.csv'.format(main_folder,file_name))
