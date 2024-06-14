import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


X_train = pd.read_csv('../Samples/Clean/Feature_Selection/X_train.csv')

print(X_train.shape)
#(573414, 35)

# Crear el modelo de clustering aglomerativo
model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)

X_train_array = X_train.to_numpy()
X_train_sampled = X_train.sample(frac=0.5, random_state=42)

model.fit(X_train_sampled)
