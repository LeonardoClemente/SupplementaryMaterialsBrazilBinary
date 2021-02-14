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
from detect_peaks import detect_peaks
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from brazil_functions import *
from brazil_constants import *
from progress.bar import Bar



brazil_stations =['Aracaju_MERRA_2', 'BeloHorizonte_MERRA_2', 'Manaus_MERRA_2']
brazil_stations =['Manaus_MERRA_2']
station = 'BeloHorizonte_MERRA_2'
full_df = pd.read_csv('{0}/{1}.csv'.format(csv_data_path, station), index_col=[0])
