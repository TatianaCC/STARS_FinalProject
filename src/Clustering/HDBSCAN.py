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
print("Cargando datasets...")
X_train = pd.read_csv('../../Samples/Clean/Feature_Selection/X_train.csv')
X_test = pd.read_csv('../../Samples/Clean/Feature_Selection/X_test.csv')
y_train = pd.read_csv('../../Samples/Clean/Feature_Selection/Y_train.csv')
y_test = pd.read_csv('../../Samples/Clean/Feature_Selection/Y_test.csv')

# endregion

X_data: pd.DataFrame = pd.concat([X_test,X_train],ignore_index=True)
y_data: pd.DataFrame = pd.concat([y_test,y_train],ignore_index=True)
print(X_data.shape)

X_data_array = X_data.to_numpy()
print("Datasets cargados!")

# region import data
X_data_with_y_PATH = '../../Samples/Clean/Testing/HDBSCAN/X_data_with_y2.csv'
X_data_with_y_predict_PATH = '../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_predict2.csv'

if not os.path.exists(X_data_with_y_PATH) or not os.path.exists(X_data_with_y_predict_PATH):
    print("X_data_with_y_PATH o X_data_with_y_predict_PATH no existe, estoy generandolo")
    model_path = "../../Models/HDBSCAN_A_minsamp-0_minclusize-5_epsilon-00_metric-man_coredist-1_genminspantree-true2.pkl"
    y_predict_path = "../../Models/HDBSCAN_A_minsamp-0_minclusize-5_epsilon-00_metric-man_coredist-1_genminspantree-true2.npy"
    if os.path.exists(model_path):
        print("El modelo ya existe")
        with open(model_path, 'rb') as file:
            clusterer = load(file)
            y_predict = np.load(y_predict_path)
    else:
        print("El modelo no existe, estoy generandolo")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, core_dist_n_jobs=11, gen_min_span_tree=True)
        y_predict = clusterer.fit_predict(X_data_array)
        dump(clusterer, open("../../Models/HDBSCAN_A_minsamp-0_minclusize-5_epsilon-00_metric-man_coredist-1_genminspantree-true2.pkl", "wb"))
    
    # Unir y_test y y_predict_test con X_test
    X_data_with_y = X_data.copy()
    X_data_with_y['cluster'] = y_data.to_numpy().ravel()
    X_data_with_y_predict = X_data.copy()
    X_data_with_y_predict['cluster_hdbscan'] = y_predict

    X_data_with_y.to_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y2.csv', index=False)
    X_data_with_y_predict.to_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_predict2.csv', index=False)
else:
    print("X_data_with_y_PATH o X_data_with_y_predict_PATH existen")

    X_data_with_y = pd.read_csv(X_data_with_y_PATH)
    X_data_with_y_predict = pd.read_csv(X_data_with_y_predict_PATH)

# endregion


unique_cluster_predict = X_data_with_y_predict.cluster_hdbscan.unique()
print(len(unique_cluster_predict))
# region graph
graphic_tester(X_data_with_y, X_data_with_y_predict, 'cluster', 'cluster_hdbscan', '_Glon', '_Glat')

# endregion
