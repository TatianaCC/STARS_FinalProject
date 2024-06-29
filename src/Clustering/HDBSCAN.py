# region Importacion librerias
import sys
import os

from sklearn.metrics import euclidean_distances
current_dir = os.getcwd()
tools_path = os.path.abspath(os.path.join(current_dir, '..', 'Tools'))
print(f"Adding '{tools_path}' to sys.path")
sys.path.append(tools_path)
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
print("Cargando datasets...")
X_train = pd.read_csv('../../Samples/Clean/Feature_Selection/X_train_machine1.csv')
X_test = pd.read_csv('../../Samples/Clean/Feature_Selection/X_test_machine1.csv')
y_train = pd.read_csv('../../Samples/Clean/Feature_Selection/Y_train_machine1.csv')
y_test = pd.read_csv('../../Samples/Clean/Feature_Selection/Y_test_machine1.csv')

# endregion

X_data: pd.DataFrame = pd.concat([X_test,X_train],ignore_index=True)
y_data: pd.DataFrame = pd.concat([y_test,y_train],ignore_index=True)
print(X_data.shape)

X_data_array = X_data.to_numpy()
print("Datasets cargados!")

# region import data
X_data_with_y_PATH = '../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_O.csv'
X_data_with_y_predict_PATH = '../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_predict_O.csv'

if not os.path.exists(X_data_with_y_PATH) or not os.path.exists(X_data_with_y_predict_PATH):
    print("X_data_with_y_PATH o X_data_with_y_predict_PATH no existe, estoy generandolo")
    model_path = "../../Models/Clustering/HDBSCAN/HDBSCAN_O.pkl"
    y_predict_path = "../../Models/Clustering/HDBSCAN/HDBSCAN_O.npy"
    if os.path.exists(model_path):
        print("El modelo ya existe")
        with open(model_path, 'rb') as file:
            clusterer = load(file)
            y_predict = np.load(y_predict_path)
    else:
        print("El modelo no existe, estoy generandolo")
        clusterer = hdbscan.HDBSCAN(min_samples=None,min_cluster_size=5, cluster_selection_epsilon=0.5, core_dist_n_jobs=11, gen_min_span_tree=True)
        y_predict = clusterer.fit_predict(X_data_array)
        dump(clusterer, open("../../Models/Clustering/HDBSCAN/HDBSCAN_O.pkl", "wb"))
    
    # Unir y_test y y_predict_test con X_test
    X_data_with_y = X_data.copy()
    X_data_with_y['cluster'] = y_data.to_numpy().ravel()
    X_data_with_y_predict = X_data.copy()
    X_data_with_y_predict['cluster_hdbscan'] = y_predict

    X_data_with_y.to_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_O.csv', index=False)
    X_data_with_y_predict.to_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_predict_O.csv', index=False)
else:
    print("X_data_with_y_PATH o X_data_with_y_predict_PATH existen")

    X_data_with_y = pd.read_csv(X_data_with_y_PATH)
    X_data_with_y_predict = pd.read_csv(X_data_with_y_predict_PATH)

# endregion


unique_cluster_predict = X_data_with_y_predict.cluster_hdbscan.unique()
print("Clusters: ",len(unique_cluster_predict))
# region graph
graphic_tester(X_data_with_y, X_data_with_y_predict, 'cluster', 'cluster_hdbscan', '_Glon', '_Glat')

# endregion
