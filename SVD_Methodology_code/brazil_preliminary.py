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

def reshape_yearly_data(df=None, t0='2002-03-01', p=40, n_years=11):

    period_matrix = []

    for i in range(0,n_years):
        if i > 0:
            # get year and increase it
            y = int(t0[0:4])
            y += 1
            t0 = '{0}'.format(y) + t0[4:]

        ind = df.index.get_loc(t0)
        timeseries = df[ind:ind+p]
        timeseries =  timeseries.T.stack().reset_index(name='new')['new'].values
        timeseries = timeseries.reshape([-1,1])
        period_matrix.append(timeseries)

    period_matrix = np.hstack(period_matrix)
    return period_matrix

def get_hulls(projections, samples_label, verbose = False):

    yes = []
    no = []
    v_yes = []
    v_no = []

    for i, v in enumerate(samples_label):
        if v == 0:
            no.append(v)
            v_no.append(projections[i,:])
        else:
            yes.append(v)
            v_yes.append(projections[i,:])

    v_no = np.vstack(v_no)
    v_yes = np.stack(v_yes)
    h_yes =ConvexHull(v_yes)
    h_no = ConvexHull(v_no)

    if verbose:
            plt.plot(projections[:,0], projections[:,1], 'o')
            for simplex in h_yes.simplices:
                plt.plot(v_yes[simplex, 0], v_yes[simplex, 1], 'k-')
            for simplex in h_no.simplices:
                plt.plot(v_no[simplex, 0], v_no[simplex, 1], 'k-', color='r')
            plt.show()

    return h_yes, h_no, v_yes, v_no

def plot_hulls(projections, h_yes, h_no, v_yes, v_no, samples_label, class_color):

    for i, label in enumerate(samples_label):
        plt.plot(projections[i,0], projections[i,1], 'o', color = class_color[label])

    for simplex in h_yes.simplices:
        plt.plot(v_yes[simplex, 0], v_yes[simplex, 1], 'k-', color='b')
        plt.plot(v_yes[simplex, 0], v_yes[simplex, 1], 'o', color='b')
    for simplex in h_no.simplices:
        plt.plot(v_no[simplex, 0], v_no[simplex, 1], 'k-', color='r')
        plt.plot(v_no[simplex, 0], v_no[simplex, 1], 'o', color='r')
    plt.show()
def plot_polygon(projections, h_yes, h_no, v_yes, v_no, samples_label, class_color):

    y = np.vstack([v_yes[h_yes.vertices,:], v_yes[h_yes.vertices[0],:]])
    n = np.vstack([v_no[h_no.vertices,:], v_no[h_no.vertices[0],:]])

    for i, label in enumerate(samples_label):
        plt.plot(projections[i,0], projections[i,1], 'o', color = class_color[label])

    plt.plot(n[:,0], n[:,1],color='r')
    plt.plot(y[:,0], y[:,1],color='b')
    plt.show()

def intersect_hulls(h_yes, h_no, v_yes, v_no):

    hull_set_no = []
    hull_set_yes = []
    indices_yes = np.unique(h_yes.simplices.flatten())
    indices_no = np.unique(h_no.simplices.flatten())

    vertices_yes = np.vstack([v_yes[h_yes.vertices,:], v_yes[h_yes.vertices[0],:]])
    vertices_no = np.vstack([v_no[h_no.vertices,:], v_no[h_no.vertices[0],:]])

    polygon_yes=Polygon(vertices_yes)
    polygon_no=Polygon(vertices_no)



    return polygon_yes.intersects(polygon_no), vertices_yes, vertices_no

def get_hull_distance( intersect_boolean, vertices_yes, vertices_no):

    if intersect_boolean:
        v_distance = 0
    else:
        h_yes_coordinates = np.vstack(vertices_yes)
        h_no_coodinates = np.vstack(vertices_no)
        v_distance = np.min(distance.cdist(h_yes_coordinates,h_no_coodinates).min(axis=1))
    return v_distance


#Repeating work in Stolerman et al.

# Read in data
full_df = pd.read_csv('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/datos_brazil.csv', skiprows=16)
del full_df['Data']
interpolate_df = full_df.interpolate()


rain  = full_df[full_df['Hora'] == 0]['Precipitacao'].to_frame()
temp  = full_df[full_df['Hora'] == 0]['Temp Comp Media'].to_frame()

full_df = pd.concat([rain, temp], axis=1).interpolate() #interpolating missing data
full_df.set_index(pd.date_range(start='2002-01-01', end='2012-12-30', inplace=True), inplace=True)

# Objective, we want to build a heatmap from SVD samples_labels

# Create a vector of t0 and p parameters

p_vector = list(range(5,91))
years = [2002]
months = list(range(6,13)) + [1,2]
days = list(range(1,32))

t0_vector = generate_date_vector(years, months, days)



index_dates = ['2002-06-01', '2002-07-01', '2002-08-01','2002-09-01', '2002-10-01',\
               '2002-11-01' ,'2002-12-01', '2002-01-01', '2002-02-01']
index_values = []
# Get indices for monthly x ticks
for d in index_dates:
    index_values.append(t0_vector.index(d))



 # Removing unexistent dates

non_existent_dates = ['2002-02-29', '2002-02-30', '2002-02-31', '2002-06-31','2002-09-31', '2002-11-31',]

for d in non_existent_dates:
    print(d)
    t0_vector.remove(d)



distance_grid = np.zeros([len(p_vector),len(t0_vector)])

# Year samples_label (epidemic = 0, non-epidemic = 1)

samples_label = [0,0,1,1,1,0,0,0,1,0,0]
class_color = {0:'r', 1:'b'}
n_years = 10
samples_label = samples_label[0:n_years]
modes = [1,2] #starts from zero

#Enter main loop
for i, p in enumerate(p_vector):
    for j, t0 in enumerate(t0_vector):
        print(p,t0)

        m = reshape_yearly_data(df=full_df, p=p, t0=t0, n_years=n_years)

        # Perform svd
        U, sigma, VT = svd(m, n_components =3, n_iter=15, random_state=None)
        projections = sigma.reshape([-1,1])*VT
        projections = projections.T

        projections = projections[:,modes]

        h_yes, h_no, v_yes, v_no = get_hulls(projections, samples_label)

        intersect_boolean, vertices_yes, vertices_no = intersect_hulls(h_yes, h_no, v_yes, v_no)

        if not intersect_boolean:
            plot_hulls(projections, h_yes, h_no, v_yes, v_no, samples_label, class_color)
            plot_polygon(projections, h_yes, h_no, v_yes, v_no, samples_label, class_color)

        distance_grid[i,j] = get_hull_distance(intersect_boolean, vertices_yes, vertices_no)

im = plt.matshow(distance_grid, cmap=plt.cm.hot, aspect='auto') # pl is pylab imported a pl
pickle.dump( [distance_grid, t0_vector, p_vector], open( "distance_grid23.p", "wb" ) )
#plt.tick_params(axis='x',which='minor',bottom='off')
plt.xticks(index_values,index_dates, rotation='90')
#plt.yticks(list(range(len(p_vector))),p_vector, rotation='90')
#plt.locator_params(axis='x', nbins=20)
plt.locator_params(axis='y', nbins=10)
plt.ylabel('P')
plt.xlabel('Start date.')
plt.show()
#Choose highest modes
