import pandas as pd
import numpy as np

# Function to remove spaces of a string
def remove_spaces(x):
    if isinstance(x, str):
        return x.replace(' ', '').strip()
    return x

# Function to remove tabulates from cells
def remove_tabs(x):
    if isinstance(x, str):
        return x.replace('\t', '')
    return x

# Function to remove duplicates from a df, in base a determinate row
# It ignore null values in the row.
def replicates_rows(df,row):
    filtered_rows = df[df[row].notna()]
    filtered_rows = filtered_rows.drop_duplicates(subset=row)
    none_rows = df[df[row].isna()]

    df_clean = pd.concat([filtered_rows, none_rows], ignore_index=True)
    
    return df_clean

#Function to rename clusters columns using NamesCatalogEquivalence
def rename_clusters(name):
    clusters_names = pd.read_csv('../Samples/Clean/NamesCatalogEquivalence.csv')
    if 'NGC' in name:
        return name
    else:
        row = clusters_names.index[clusters_names['Actual Names'] == name]
        if not row.empty and clusters_names.at[row[0], 'NGC_names'] is not None:
            return clusters_names.at[row[0], 'NGC_names']
        else:
            return name
        
#Function to get columns with % of null values over a gived top
def nulls_columns(df, n_pc):
    total_values_per_column = df.shape[0]
    
    null_values_per_column = df.isnull().sum()
    percentage_null_per_column = (null_values_per_column / total_values_per_column) * 100
    percentage_null_per_column.sort_values(ascending=False)
    cols_to_drop_nul_pc = percentage_null_per_column[percentage_null_per_column>n_pc]

    return cols_to_drop_nul_pc 

#Function to get the number of null values in each cluster
def nulls_in_cluster(df):
    nulos_por_cluster = df.groupby('Cluster')['Plx'].apply(lambda x: x.isnull().sum())
    nulos_por_cluster_total = df.groupby('Cluster').size()
    nulos_por_cluster_nulos = nulos_por_cluster[nulos_por_cluster > 0]

    df_t = nulos_por_cluster_total.reset_index()
    df_n = nulos_por_cluster_nulos.reset_index()

    df_t.columns = ['Cluster', 'nulos_t']
    df_n.columns = ['Cluster', 'nulos_n']

    df_merged = pd.merge(df_t, df_n, on='Cluster',how='inner',validate='1:1')

    df_merged['n_percentage'] = df_merged.apply(lambda row: 100* row['nulos_n'] / row['nulos_t'] if row['nulos_t'] != 0 else float('inf'), axis=1)
    df_merged_sorted = df_merged.sort_values(by='n_percentage', ascending=False)

    return df_merged_sorted 
    print(df_merged_sorted.sort_values(by='nulos_t', ascending=False))

def split_clusters(df):
    grouped = df.groupby('Cluster_O')
    for cluster_name, group in grouped:
        filename = f"{cluster_name}.csv"
        group.to_csv('../Samples/Clean/Subsets/'+filename, index=False)
        