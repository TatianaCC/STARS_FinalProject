######################################################################
#           STARS (Stellar association recognition system)           #
######################################################################
"""
    Ensemble of unsupervised and supervised classification models.

PART I: Finding associations with an unsupervised clustering model
We chose HDBSCAN for four reasons:
- It can work with very large data samples.
- It can identify outliers or noise.
- It is not necessary to specify the number of clusters to find.
- It can find non-spherical clusters.

PART II: Checking coherence with a supervised classification model
We chose Random Forest for five reasons:
- Less overfitting and high precision.
- It can work with high-dimensional data and estimate feature importance.
- It does not need as much optimization as other models.
- Easy to implement.
- Robust against noise.

PART III: Ensemble to optimize HDBSCAN
We chose Bayesian optimization for four reasons:
- Scalability.
- Suitable for Costly Functions.
- It is more efficient than grid search or random search because it requires 
  fewer evaluations.
- Focusing on Promising Regions.
"""

#·······························IMPORTS·······························#
from scipy.optimize._optimize import OptimizeResult
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

space_hdbscan = [
            Integer(5, 20, name='min_cluster_size'),
            Integer(0, 20, name='min_samples'),
            Real(0.0, 0.5, name='cluster_selection_epsilon')
]

class STARS:
    def __init__(self):
        self.data_csv: str = ""
        self.x_all = pd.DataFrame()
        self.x_data_array = np.array([])
        self.space_hdbscan = [
            Integer(5, 20, name='min_cluster_size'),
            Integer(0, 20, name='min_samples'),
            Real(0.0, 0.5, name='cluster_selection_epsilon')
        ]

    def main(self):
        return None
    
    @use_named_args(dimensions=space_hdbscan)
    def objective_hdbscan(self,params, *args):
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(self.x_data_array)
        
        # Not taking into account noise before calculating the score
        valid_labels = labels[labels != -1]
        if len(valid_labels) > 1:
            score = silhouette_score(self.x_data_array[labels != -1], valid_labels,random_state=42)
        else:
            score = -1  # If there are only one cluster or noise
        
        # Return negative of combined score (to minimize)
        return -(score + np.mean(labels != -1))
       
    def tatis_while(self):
    
        max_iter = 10
        iter_count = 0
        coherence = 0
        while coherence < 0.95 and iter_count < max_iter:
            # HDBSCAN optimization
            res_hdbscan: OptimizeResult | None = gp_minimize(func=self.objective_hdbscan, dimensions=self.space_hdbscan, n_calls=50, random_state=42)
            if res_hdbscan is not None:
                best_params_hdbscan = dict(zip([dim.name for dim in self.space_hdbscan], res_hdbscan.x))
            
                # Instantiate HDBSCAN with best parameters
                clusterer = hdbscan.HDBSCAN(**best_params_hdbscan)
                clusters_hdbscan = clusterer.fit_predict(self.x_data_array)
                
                # Store clusters back into X_all and identify noise
                x_cluster = self.x_all.copy(deep=True)
                x_cluster['cluster_hdbscan'] = clusters_hdbscan
                data_noise = self.x_all[self.x_all['cluster_hdbscan'] == -1]

                # Calculate number of clusters and minimal cluster size
                unique_clusters, counts_clusters = np.unique(clusters_hdbscan, return_counts=True)
                num_clusters = len(unique_clusters) - 1  # Exclude noise cluster
                min_cluster_size = min(counts_clusters[counts_clusters > 1])  # Exclude noise

                # Split data for Random Forest training
                x_train, x_test, y_train, y_test = train_test_split(self.x_data_array, clusters_hdbscan, test_size=0.3, random_state=42)
                
                # Train Random Forest with number of clusters as estimators
                rf = RandomForestClassifier(n_estimators=num_clusters, min_samples_split=min_cluster_size, random_state=42)
                rf.fit(x_train, y_train)
                
                # Predict clusters for all data
                y_pred_full = rf.predict(self.x_data_array)
                x_cluster['cluster_randomforest'] = y_pred_full

                # Calculate coherence
                coherence = np.mean(y_pred_full == clusters_hdbscan)
                print(f"Iteration {iter_count + 1} - Coherence: {coherence * 100:.2f}%")
                
                iter_count += 1

            print(f"Final coherence: {coherence * 100:.2f}%")

    #································TOOLS·······························#
    def graphic(self, df, groupby, x_axe, y_axe, save_path=None):
        df['Index'] = df.index
        cluster_means_real = df.groupby(groupby).agg({
            x_axe: 'mean',
            y_axe: 'mean',
            'Index': 'count',
        }).reset_index()

        plt.figure(figsize=(20, 8))
        sns.scatterplot(cluster_means_real, x=x_axe, y=y_axe, size='Index', alpha=0.4, sizes=(10,5000), legend=False,color='#4c72b0')
        if save_path:
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.show()

    def report(self,_best_params_hdbscan, _num_clusters, _counts_clusters, _min_cluster_size, _coherence,_iter_count):
        readme_content = f"""
        HDBSCAN Hyperparameters:
        - min_cluster_size: {_best_params_hdbscan['min_cluster_size']}
        - min_samples: {_best_params_hdbscan['min_samples']}
        - cluster_selection_epsilon: {_best_params_hdbscan['cluster_selection_epsilon']:.2f}

        Number of Clusters Found: {_num_clusters}
        Cluster Sizes: {_counts_clusters.tolist()}

        Random Forest Hyperparameters:
        - n_estimators: {_num_clusters}
        - min_samples_split: {_min_cluster_size}

        Coherence: {_coherence * 100:.2f}%
        Number of Iterations: {_iter_count}
        """

        with open('README.txt', 'w') as f:
            f.write(readme_content)
    
    def todolodemas(self):

        print("Loaded dataset")

        #···················BAYESIAN HDBSCAN OPTIMIZATION····················#
        # Define search space

        

      

        X_cluster = pd.DataFrame()
        data_noise = pd.DataFrame()
        #·····························MAIN······························#
        

        #···························SAVE AND REPORT····························#
        # Save models
        hdbscan_model_path = f"hdbscan_model_{best_params_hdbscan['min_cluster_size']}_{best_params_hdbscan['min_samples']}_{best_params_hdbscan['cluster_selection_epsilon']:.2f}.pkl"
        rf_model_path = f"random_forest_model_{num_clusters}_{min_cluster_size}.pkl"

        # Save samples
        X_cluster.to_csv('data_with_clusters.csv', index=False)
        data_noise.to_csv('data_noise.csv', index=False)
        pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
        pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
        pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

        # Save graph
        graphic(X_cluster, 'cluster_hdbscan', '_Glon', '_Glat', save_path=None)

        # Write report
        report(best_params_hdbscan, num_clusters, counts_clusters, min_cluster_size, coherence,iter_count)

    def doyourthing(self, filepath: str, db_id:int | None) -> None:
    #·····························LOAD DATA······························#
        self.data_csv = filepath
        self.x_all = pd.read_csv(self.data_csv)   
        self.x_data_array = self.x_all.to_numpy()
        print(db_id,str(filepath))
        self.tatis_while()

if __name__ == "__main__":
    instancia = STARS()
    instancia.objective_hdbscan()