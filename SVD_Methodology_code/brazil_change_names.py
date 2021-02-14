import pandas as pd

folder = '/Users/leonardo/Desktop/FORECASTING/SARAH/'
output_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV/'
files = ['Aracaju_MERRA_2.csv', 'BeloHorizonte_MERRA_2.csv', 'Manaus_MERRA_2.csv']

for f in files:
    df = pd.read_csv(folder+f, index_col=[0])
    df = df[['temp', 'precip']]
    df.columns = ['Temp Comp Media', 'Precipitacao']
    df.to_csv(output_folder+f)
