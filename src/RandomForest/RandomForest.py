import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pickle import dump, load

# Load data
df = pd.read_csv('../../Samples/Clean/Testing/HDBSCAN/X_data_with_y_predict_Q.csv')
y_hdbscan = df['cluster_hdbscan']
x_full = df.drop(columns=['cluster_hdbscan']).copy(deep=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(x_full, y_hdbscan, test_size=0.2, random_state=42)

# Train random forest
n_cluster = len(y_hdbscan.unique().tolist())
rf = RandomForestClassifier(n_estimators=n_cluster, random_state=42, min_samples_split=20)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Accuracy
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Predict in full dataset
y_pred_full = rf.predict(x_full)

# Coherence
coherencia = np.mean(y_pred_full == y_hdbscan)
print(f"Coherencia: {coherencia * 100:.2f}%")

# Save
df_final = df.copy(deep = True)
df_final['cluster_RF'] = y_pred_full
df_final.to_csv('../../Samples/Clean/Final/STARTSOutput.csv', index=False)
dump(rf, open("../../Models/Random_Forest/RF_Q.pkl", "wb"))