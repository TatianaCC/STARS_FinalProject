# region Importacion librerias
import sys
import os
current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'Tools')))
import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pickle import dump, load
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from tools import graphic_tester


# endregion

# region Importacion datasets
X_train = pd.read_csv('../../Samples/Clean/Feature_Selection/X_train.csv')
X_test = pd.read_csv('../../Samples/Clean/Feature_Selection/X_test.csv')
y_train = pd.read_csv('../../Samples/Clean/Feature_Selection/Y_train.csv')
y_test = pd.read_csv('../../Samples/Clean/Feature_Selection/Y_test.csv')

# endregion

# region import data
X_data: pd.DataFrame = X_test.add(X_train)
y_data: pd.DataFrame = y_test.add(y_train)

X_data_array = X_data.to_numpy()
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=11, gen_min_span_tree=True)
y_predict = clusterer.fit_predict(X_data_array)

# Unir y_test y y_predict_test con X_test
X_data_with_y = X_data.copy()
X_data_with_y['cluster'] = y_data.to_numpy().ravel()
X_data_with_y_predict = X_data.copy()
X_data_with_y_predict['cluster_hdbscan'] = y_predict

dump(clusterer, open("../../Models/HDBSCAN_A_minsamp-0_minclusize-5_epsilon-00_metric-man_coredist-1_genminspantree-true.pkl", "wb"))
X_data_with_y.to_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y.csv', index=False)
X_data_with_y_predict.to_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_predict.csv', index=False)
# endregion

# region graph
graphic_tester(X_data_with_y, X_data_with_y_predict, 'cluster', 'cluster_hdbscan', '_Glon', '_Glat')

# endregion