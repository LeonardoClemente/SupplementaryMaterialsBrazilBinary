import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import copy
import time
from sklearn.utils.extmath import randomized_svd as svd
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from scipy.spatial import distance
import pickle



# Get periods

def generate_date_vector(years, months, days):

    date_vector = []
    for i in years:
        for j in months:
            for k in days:
                if k < 10 and j < 10:
                    date_vector.append('{0}-0{1}-0{2}'.format(i, j, k))
                elif k >= 10 and j < 10:
                    date_vector.append('{0}-0{1}-{2}'.format(i, j, k))
                elif k < 10 and j >= 10:
                    date_vector.append('{0}-{1}-0{2}'.format(i, j, k))
                elif k >= 10 and j >= 10:
                    date_vector.append('{0}-{1}-{2}'.format(i, j, k))

    return date_vector

def reshape_yearly_data(df=None, t0='2002-03-01', p=40, n_years=11, normalize=True):

    period_matrix = []
    y = int(t0[0:4])
    for i in range(0,n_years):
        t = '{0}'.format(y+i) + t0[4:]

        ind = df.index.get_loc(t)
        timeseries = df[ind:ind+p]

        timeseries =  timeseries.T.stack().reset_index(name='new')['new'].values
        timeseries = timeseries.reshape([-1,1])
        period_matrix.append(timeseries)

    period_matrix = np.hstack(period_matrix)
    if normalize:
        period_matrix = standardize_matrix(period_matrix)
    return period_matrix


def reshape_yearly_data_stolerman(df=None, t0='2002-03-01', p=40, n_years=11, normalize=True):

    period_matrix = []
    timeseries_list = []
    indices = []
    y = int(t0[0:4])
    for i in range(0,n_years):
        t = '{0}'.format(y+i) + t0[4:]

        ind = df.index.get_loc(t)
        timeseries_list.append(df[ind:ind+p])

    stacked_timeseries = pd.concat(timeseries_list, axis=0)
    if normalize:
        stacked_timeseries = standardize_df(stacked_timeseries)

    for i in range(0,n_years):

        t = '{0}'.format(y+i) + t0[4:]

        ind = stacked_timeseries.index.get_loc(t)
        timeseries = stacked_timeseries[ind:ind+p]
        timeseries =  timeseries.T.stack().reset_index(name='new')['new'].values
        timeseries = timeseries.reshape([-1,1])
        period_matrix.append(timeseries)

    period_matrix = np.hstack(period_matrix)
    return period_matrix

def standardize_df(df=None):
    #Normalize each column of a dataframe
    df_names = list(df)

    for name in df_names:
        mu = df[name].values.mean()
        sigma = df[name].values.std()
        df[name] = (df[name]-mu)/sigma
    return df

def standardize_matrix(matrix=None):

    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0)
    matrix -= mu
    matrix /= sigma

    return matrix


def get_hulls(projections, year_epidemic_classification, verbose = False):

    yes = []
    no = []
    coordinates_1 = []
    coordinates_2 = []

    for i, v in enumerate(year_epidemic_classification):
        if v == 0:
            no.append(v)
            coordinates_2.append(projections[i,:])
        else:
            yes.append(v)
            coordinates_1.append(projections[i,:])

    coordinates_2 = np.vstack(coordinates_2)
    coordinates_1 = np.stack(coordinates_1)
    hull_1 =ConvexHull(coordinates_1)
    hull_2 = ConvexHull(coordinates_2)

    if verbose:
            plt.plot(projections[:,0], projections[:,1], 'o')
            for simplex in hull_1.simplices:
                plt.plot(coordinates_1[simplex, 0], coordinates_1[simplex, 1], 'k-')
            for simplex in hull_2.simplices:
                plt.plot(coordinates_2[simplex, 0], coordinates_2[simplex, 1], 'k-', color='r')
            plt.show()

    return hull_1, hull_2, coordinates_1, coordinates_2

def plot_hulls(projections, hull_1, hull_2, coordinates_1, coordinates_2, year_epidemic_classification, class_color):

    for i, label in enumerate(year_epidemic_classification):
        plt.plot(projections[i,0], projections[i,1], 'o', color = class_color[label])

    for simplex in hull_1.simplices:
        plt.plot(coordinates_1[simplex, 0], coordinates_1[simplex, 1], 'k-', color='b')
        plt.plot(coordinates_1[simplex, 0], coordinates_1[simplex, 1], 'o', color='b')
    for simplex in hull_2.simplices:
        plt.plot(coordinates_2[simplex, 0], coordinates_2[simplex, 1], 'k-', color='r')
        plt.plot(coordinates_2[simplex, 0], coordinates_2[simplex, 1], 'o', color='r')
    plt.show()
def plot_polygon(projections, hull_1, hull_2, coordinates_1, coordinates_2, year_epidemic_classification, class_color):

    y = np.vstack([coordinates_1[hull_1.vertices,:], coordinates_1[hull_1.vertices[0],:]])
    n = np.vstack([coordinates_2[hull_2.vertices,:], coordinates_2[hull_2.vertices[0],:]])

    for i, label in enumerate(year_epidemic_classification):
        plt.plot(projections[i,0], projections[i,1], 'o', color = class_color[label])

    plt.plot(n[:,0], n[:,1],color='r')
    plt.plot(y[:,0], y[:,1],color='b')
    plt.show()

def intersect_hulls(hull_1, hull_2, coordinates_1, coordinates_2):

    vertices_1 = np.vstack([coordinates_1[hull_1.vertices,:], coordinates_1[hull_1.vertices[0],:]])
    vertices_2 = np.vstack([coordinates_2[hull_2.vertices,:], coordinates_2[hull_2.vertices[0],:]])

    polygon_1=Polygon(vertices_1)
    polygon_2=Polygon(vertices_2)

    return polygon_1.intersects(polygon_2), vertices_1, vertices_2

def get_hull_distance( intersect_boolean, vertices_1, vertices_2):

    if intersect_boolean:
        v_distance = 0
    else:
        hull_1_coordinates = np.vstack(vertices_1)
        hull_2_coodinates = np.vstack(vertices_2)
        v_distance = np.min(distance.cdist(hull_1_coordinates,hull_2_coodinates).min(axis=1))
    return v_distance


#Repeating work in Stolerman et al.


# Files and folders
heatmap_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/HEATMAPS'
csv_data_path = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'

# Read in data
full_df = pd.read_csv('{0}/ACARAJU.csv'.format(csv_data_path), skiprows=16)


#del full_df['Data']
#full_df.set_index(full_df['Data'], inplace=True)


interpolate_df = full_df.interpolate()

rain  = full_df[full_df['Hora'] == 0]['Precipitacao'].to_frame()
temp  = full_df[full_df['Hora'] == 0]['Temp Comp Media'].to_frame()

full_df = pd.concat([rain, temp], axis=1).interpolate() #interpolating missing data

all_dates = pd.date_range(start='2001-01-01', end='2012-12-31').strftime("%Y-%m-%d")

full_df.set_index(all_dates, inplace=True)

#full_df.set_index(pd.date_range(start='2001-01-01', end='2012-12-31'), inplace=True)

# Objective, we want to build a heatmap from SVD samples_labels

# Create a vector of t0 and p parameters
normalize = True
p_vector = list(range(10,100))


years = [2001]
months = list(range(6,13))
days = list(range(1,32))

t0_vector = generate_date_vector(years, months, days) + generate_date_vector([2002], list(range(1,3)), days)

index_dates = ['2001-06-01', '2001-07-01', '2001-08-01','2001-09-01', '2001-10-01',\
               '2001-11-01' ,'2001-12-01', '2002-01-01', '2002-02-01']
non_existent_dates = ['2002-02-29', '2002-02-30', '2002-02-31', '2001-06-31','2001-09-31', '2001-11-31',]


index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))



 # Removing unexistent dates

for d in non_existent_dates:
    print(d)
    t0_vector.remove(d)



distance_grid = np.zeros([len(p_vector),len(t0_vector)])

# year_epidemic_classification (epidemic = 0, non-epidemic = 1)

year_epidemic_classification = [0,0,1,1,1,0,0,0,1,0,0]
class_color = {0:'r', 1:'b'}
n_years = 11
year_epidemic_classification = year_epidemic_classification[0:n_years]
modes = [0,1] #starts from zero

#Enter main loop
for i, p in enumerate(p_vector):
    for j, t0 in enumerate(t0_vector):
        print(p,t0)

        X = SVDC_reshape_yearly_data_stolerman(df, t0=t0, p=p, years, normalize=True)

        # Perform svd
        U, sigma, VT = svd(X, n_components =3, n_iter=15, random_state=None)
        projections = sigma.reshape([-1,1])*VT
        projections = projections.T

        projections = projections[:,modes]

        hull_1, hull_2, coordinates_1, coordinates_2 = get_hulls(projections, year_epidemic_classification)

        intersect_boolean, vertices_1, vertices_2 = intersect_hulls(hull_1, hull_2, coordinates_1, coordinates_2)

        #if not intersect_boolean:
        #    plot_hulls(projections, hull_1, hull_2, coordinates_1, coordinates_2, year_epidemic_classification, class_color)
        #    plot_polygon(projections, hull_1, hull_2, coordinates_1, coordinates_2, year_epidemic_classification, class_color)

        distance_grid[i,j] = get_hull_distance(intersect_boolean, vertices_1, vertices_2)



data = pickle.dump([distance_grid, t0_vector, p_vector], open("{0}/clemente{1}_norm{2}.p".format(heatmap_path, modes, normalize), "wb" ))
im = plt.matshow(distance_grid, cmap=plt.cm.hot, aspect='auto', origin='lower') # pl is pylab imported a pl

plt.colorbar()
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,index_dates, rotation='90')
plt.yticks([0,10,20,30,40,50,60,70,80,90], [10,20,30,40,50,60,70,80,90,100])
#plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=10)
plt.ylabel('P')
plt.xlabel('Start date.')
plt.show()
#Choose highest modes
