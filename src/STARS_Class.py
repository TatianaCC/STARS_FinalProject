import numpy as np
import pandas as pd
from scipy.optimize._optimize import OptimizeResult
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import hdbscan
from pickle import dump
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, Any, List
import os
import zipfile

class STARS:
    def __init__(self, data_file_path,db_id,email) -> None:
        print("init")
        # Init variables, counter and space of hyperparameters
        self.path_files : str = "C:/Users/milser/Documents/Trasteo_4geeks/STARS_FinalProject/data/Streamlit_data/results/"
        self.path_files += str(db_id)+'/'
        self.db_id :str = str(db_id)
        if not os.path.exists(self.path_files):
            os.makedirs(self.path_files)
        self.email = email
        self.max_iter = 10
        self.coherence_threshold = 0.95

        self.iter_count = 0
        self.coherence = 0.0

        self.space_hdbscan = [
            Integer(5, 22, name='min_cluster_size'),
            Integer(1, 20, name='min_samples'),
            Real(0.0, 0.5, name='cluster_selection_epsilon')
        ]
        
        # Load data
        print("loading...")
        self.x_all = self.load_data(data_file_path)
        self.x_data_array = self.x_all.to_numpy()
        print("optimize_loop")
        self.optimize_loop()
        
    # Function for load data
    def load_data(self,data_file_path) -> pd.DataFrame:
        return pd.read_csv(data_file_path)

    # Function objective for optimization
    def evaluate_hdbscan(self, params: List[Any]) -> float:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params[0],
            min_samples=params[1],
            cluster_selection_epsilon=params[2],
            core_dist_n_jobs=11
        )
        labels = clusterer.fit_predict(self.x_data_array)
        
        # Not taking into account noise before calculating the score
        valid_labels = labels[labels != -1]
        if len(valid_labels) > 1:
            score = float(silhouette_score(self.x_data_array[labels != -1], valid_labels, random_state=42))
        else:
            score = -1.0  # If there are only one cluster or noise

        return -score  # Return negative of score

    # Function for get best params of HDBSCAN optimization
    def optimize_hdbscan(self) -> Dict[str, Any]:
        res_hdbscan = gp_minimize(self.evaluate_hdbscan, self.space_hdbscan, n_calls=50, random_state=42)
        
        if res_hdbscan is not None and hasattr(res_hdbscan, 'x'):
            best_params_hdbscan = dict(zip([dim.name for dim in self.space_hdbscan], res_hdbscan.x))
        else:
            # Valores predeterminados en caso de que gp_minimize no devuelva un resultado válido
            best_params_hdbscan = {dim.name: 0 for dim in self.space_hdbscan}
        
        return best_params_hdbscan

    # Function for build and train Random Forest
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, num_clusters: int, min_cluster_size: int) -> RandomForestClassifier:
        rf = RandomForestClassifier(n_estimators=num_clusters, min_samples_split=min_cluster_size, random_state=42)
        rf.fit(X_train, y_train)
        return rf

    # Loop function
    def optimize_loop(self) -> None:
        best_params_hdbscan = {}
        num_clusters = 0
        min_cluster_size = 0
        counts_clusters = np.array([])
        clusters_hdbscan = np.array([])
        X_train = np.array([])
        X_test_df = pd.DataFrame()
        y_train = np.array([])
        y_test = np.array([])
        clusterer = hdbscan.HDBSCAN()
        rf: RandomForestClassifier = RandomForestClassifier(random_state=42)
        print("pre-while")
        while self.coherence < self.coherence_threshold and self.iter_count < self.max_iter:
            print("loop: ", self.iter_count)
            # HDBSCAN optimization
            best_params_hdbscan: Dict[str, Any] = self.optimize_hdbscan()
            
            # HDBSCAN with best parameters
            clusterer = hdbscan.HDBSCAN(**best_params_hdbscan)
            clusters_hdbscan = clusterer.fit_predict(self.x_data_array)
            print("predicted...")
            # Calculate number of clusters and minimal cluster size
            unique_clusters, counts_clusters = np.unique(clusters_hdbscan, return_counts=True)
            num_clusters = max(len(unique_clusters) - 1, 1)  # Exclude noise cluster
            min_cluster_size = min(counts_clusters[counts_clusters > 1])  # Exclude noise
            print("unique_clusters: "+str(num_clusters))

            # Split data for Random Forest training
            X_train, X_test, y_train, y_test = train_test_split(self.x_all, clusters_hdbscan, test_size=0.3, random_state=42)
            
            # Train Random Forest with number of clusters and it size as estimators
            rf: RandomForestClassifier = self.train_random_forest(X_train, y_train, num_clusters, min_cluster_size)
            # Predict clusters for test data
            y_pred = rf.predict(X_test)
            print("rf predicted...")
            X_test_df = pd.DataFrame(X_test, columns=self.x_all.columns)
            X_test_df['cluster_randomforest'] = y_pred

            # Calculate coherence
            self.coherence = np.mean(y_pred == y_test)
            print(f"Iteration {self.iter_count + 1} - Coherence: {self.coherence * 100:.2f}%")
            
            self.iter_count += 1
            
        # Store clusters back into X_all and identify noise
        self.x_all['cluster_hdbscan'] = clusters_hdbscan

        print(f"Final coherence: {self.coherence * 100:.2f}%")
        
        # Save models and results
        self.save_results(best_params_hdbscan, num_clusters, min_cluster_size, counts_clusters, X_train, X_test_df, y_train, y_test, clusterer, rf)
    
    def comprimir_carpeta(self,carpeta_path:str,db_id:str) -> str:
        # Nombre del archivo ZIP que se va a crear
        
        # Crear un archivo ZIP
        with zipfile.ZipFile(carpeta_path + db_id + "_"+ self.email +".zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(carpeta_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, carpeta_path))
                    
        return carpeta_path + db_id +".zip"   
    # Function for save results
    def save_results(self, best_params_hdbscan: Dict[str, Any], num_clusters: int, min_cluster_size: int, counts_clusters: np.ndarray, 
                     X_train: np.ndarray, X_test_df: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray, 
                     clusterer: hdbscan.HDBSCAN, rf: RandomForestClassifier) -> None:
        # Save models
        hdbscan_model_path = f"hdbscan-min_cluster_size_{best_params_hdbscan['min_cluster_size']}-min_samples_{best_params_hdbscan['min_samples']}-cluster_selection_epsilon_{best_params_hdbscan['cluster_selection_epsilon']:.2f}.pkl"
        dump(clusterer, open(hdbscan_model_path, "wb"))
        rf_model_path = f"rf-num_clusters_{num_clusters}-min_cluster_size_{min_cluster_size}.pkl"
        dump(rf, open(rf_model_path, "wb"))
        
        # Save samples
        self.x_all.to_csv(self.path_files+'data_with_clusters.csv', index=False)
        pd.DataFrame(X_train).to_csv(self.path_files+'X_train.csv', index=False)
        X_test_df.to_csv(self.path_files+'X_test.csv', index=False)
        pd.DataFrame(y_train).to_csv(self.path_files+'y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv(self.path_files+'y_test.csv', index=False)

        # Graph
        self.x_all['Index'] = self.x_all.index
        cluster_means_real = self.x_all.groupby('cluster_hdbscan').agg({
            '_Glon': 'mean',
            '_Glat': 'mean',
            'Index': 'count',
        }).reset_index()

        plt.figure(figsize=(20, 8))
        sns.scatterplot(data=cluster_means_real, x='_Glon', y='_Glat', size='Index', alpha=0.4, sizes=(10,5000), legend=False, color='#4c72b0')
        plt.savefig(self.path_files+'HDBSCAN_clusters.svg', format='svg', bbox_inches='tight')

        archivo_zip: str = self.comprimir_carpeta(self.path_files,self.db_id)
        print(f"Archivo ZIP creado: {archivo_zip}")

        # Write report
        readme_content = f"""
        HDBSCAN Hyperparameters:
        - min_cluster_size: {best_params_hdbscan['min_cluster_size']}
        - min_samples: {best_params_hdbscan['min_samples']}
        - cluster_selection_epsilon: {best_params_hdbscan['cluster_selection_epsilon']:.2f}

        Number of Clusters Found: {num_clusters}
        Cluster Sizes: {counts_clusters.tolist()}

        Random Forest Hyperparameters:
        - n_estimators: {num_clusters}
        - min_samples_split: {min_cluster_size}

        Coherence: {self.coherence * 100:.2f}%
        Number of Iterations: {self.iter_count}
        """

        with open('README.txt', 'w') as f:
            f.write(readme_content)

    

# Esto es lo que hay que poner en app para hacer la instancia de la clase y que arranque
# stars = STARS('data.csv')
