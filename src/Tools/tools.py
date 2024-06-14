import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn import metrics

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
    clusters_names = pd.read_csv('../Samples/Auxiliar/NamesCatalogEquivalence.csv')
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

# Function to find outliers by pmRA and pmDE
def pm_analysis(k, cluster):
    df = pd.read_csv('../Samples/Clean/Subsets/'+cluster+'.csv')
    # Outliers
    var =  ['pmRA','pmDE']
    limits = {}
    for v in range(2):
        mean = np.mean(df[var[v]])
        sigma = np.std(df[var[v]])
        upper_l = mean + k * sigma
        lower_l = mean - k * sigma    
        limits[var[v]] = [lower_l, upper_l]
 
    mask_pmRA = (df['pmRA'] >= limits['pmRA'][1]) | (df['pmRA'] <= limits['pmRA'][0])
    mask_pmDE = (df['pmDE'] >= limits['pmDE'][1]) | (df['pmDE'] <= limits['pmDE'][0])

    valid = df[~(mask_pmRA | mask_pmDE)]
    print('N_Stars: '+str(df.shape[0])+'\nN_Valid: '+str(valid.shape[0])+'\n'+str(round((valid.shape[0]/df.shape[0])*100,2))+'%')

    # Bins
    bins=20    
    data_minRA = min(df.pmRA.min(), valid.pmRA.min())
    data_maxRA = max(df.pmRA.max(), valid.pmRA.max())        
    bin_widthRA = (data_maxRA-data_minRA)/bins
    data_minDE = min(df.pmDE.min(), valid.pmDE.min())
    data_maxDE = max(df.pmDE.max(), valid.pmDE.max())        
    bin_widthDE = (data_maxDE-data_minDE)/bins

    sns.set(style="white")
    background_image = plt.imread('../images/fondo.jpg')
    fig, axs = plt.subplots(1, 5, figsize=(15, 6), 
                            gridspec_kw={"width_ratios": [1, 9, 1, 1, 9]})
    # pmRA boxplot
    sns.boxplot(y=df['pmRA'], ax=axs[0])
    axs[0].set_ylabel('pmRA', fontsize=20)    

    # pmRA histogram
    sns.histplot(df, y='pmRA', bins=bins, ax=axs[1], element="bars", fill=True, color='white', alpha=0.5, binwidth=bin_widthRA, binrange=(data_minRA,data_maxRA))
    sns.histplot(valid, y='pmRA', bins=bins, ax=axs[1], element="bars", fill=True, color='orange', alpha=0.5, binwidth=bin_widthRA, binrange=(data_minRA,data_maxRA))
    axs[1].imshow(background_image, aspect='auto', extent=axs[1].get_xlim() + axs[1].get_ylim(), alpha=0.9)
    axs[1].set_ylim(data_minRA,data_maxRA)

    # pmDE boxplot
    sns.boxplot(y=df['pmDE'], ax=axs[3])
    axs[3].set_ylabel('pmDE', fontsize=20)

    # pmDE histogram
    sns.histplot(df, y='pmDE', bins=20, ax=axs[4], element="bars", fill=True, color='white', alpha=0.5, binwidth=bin_widthDE, binrange=(data_minDE,data_maxDE))
    sns.histplot(valid, y='pmDE', bins=20, ax=axs[4], element="bars", fill=True, color='orange', alpha=0.5, binwidth=bin_widthDE, binrange=(data_minDE,data_maxDE))
    axs[4].imshow(background_image, aspect='auto', extent=axs[4].get_xlim() + axs[4].get_ylim(), alpha=0.9)
    axs[4].set_ylim(data_minDE,data_maxDE)

    for i in range(5):
        if i == 2:
            axs[i].axis('off')
        elif i == 0 or i == 3:
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        elif i == 1 or i == 4:
            axs[i].set_ylabel('')
            axs[i].set_xlabel('')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['left'].set_visible(True)
            axs[i].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
            axs[i].tick_params(axis='x', which='both', left=False, right=False, labelleft=False)
            axs[i].set_xticklabels([])
    
    fig.suptitle('Movimientos propios de '+cluster, fontsize=16)#
    fig.patch.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.1) 
    plt.show()

    return valid


def explore(data_frame: pd.DataFrame):
    """
    Explores a pandas DataFrame and prints information about its characteristics.

    This function `prints` useful `information` about the provided pandas DataFrame,
    including the number of `rows` and `columns`, the count of `null` and `non-null` values,
    the `data type` of each column, and whether each column is `categorical` or `numerical`.

    Parameters::

        data_frame (pandas.DataFrame): The pandas DataFrame to be explored.

    """
    data_frame_ = data_frame.copy()
    
    # Get shape
    num_rows_, num_columns_ = data_frame_.shape
    print('Rows:', num_rows_)
    print('Columns:', num_columns_)

    # Get null and type info
    column_data_ = pd.DataFrame({
        'Non-Null Count': data_frame_.count(),
        'Null Count': data_frame_.isnull().sum(),
        'Nan Count': data_frame_.isna().sum(),
        'Data Type': data_frame_.dtypes
    }) 
    # Add if a variable is categorical or numerical  
    column_data_['Data Category'] = data_frame_.apply(get_column_type)
    print(tabulate(column_data_, headers='keys', tablefmt='pretty'))
     

def get_column_type(series: str):
    """
    Determines the column type of a pandas series.

    This function takes a pandas series as input and determines whether the data
    in the series are `Numerical` or `categorical`.

    Parameters::

        series (pandas.Series): The pandas series from which the column type will be determined.

    Returns::
    
        str: 'Numerical': If they are Numerical.
             'Categorical': If they are categorical.
    """
    series_ = series.copy()
    
    if pd.api.types.is_numeric_dtype(series_):
        return 'Numerical'
    else:
        return 'Categorical'
    
def graphic_tester(df_real, df_model, groupby_real, groupby_model, x, y):
    df_real['Index'] = df_real.index
    cluster_means_real = df_real.groupby(groupby_real).agg({
        x: 'mean',
        y: 'mean',
        'Index': 'count',
    }).reset_index()

    df_model['Index'] = df_model.index
    cluster_means_model = df_model.groupby(groupby_model).agg({
        x: 'mean',
        y: 'mean',
        'Index': 'count',
    }).reset_index()

    # Graph
    plt.figure(figsize=(20, 8))
    sns.scatterplot(cluster_means_real, x=x, y=y, size='Index', alpha=0.4, sizes=(10,5000), legend=False,color='#4c72b0')
    sns.scatterplot(cluster_means_model, x=x, y=y, size='Index', alpha=0.5, sizes=(10,5000), legend=False, color='#dd8452')
    plt.show()

def clustering_metrics(X, labels_true, labels_pred):
    # Internal metrics
    #silhouette = silhouette_score(X, labels_pred)
    #davies_bouldin = davies_bouldin_score(X, labels_pred)
    #calinski_harabasz = calinski_harabasz_score(X, labels_pred)

    # External metrics
    homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
    completeness = metrics.completeness_score(labels_true, labels_pred)
    v_measure = metrics.v_measure_score(labels_true, labels_pred)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)

    # Evaluation criteria
    def rate_metric(value, metric_name):
        if metric_name in ['Silhouette Score', 'Homogeneity', 'Completeness', 'V-Measure', 'ARI', 'AMI']:
            if value >= 0.8:
                return '★★★★★'
            elif value >= 0.6:
                return '★★★★'
            elif value >= 0.4:
                return '★★★'
            elif value >= 0.2:
                return '★★'
            else:
                return '★'
        elif metric_name == 'Davies-Bouldin Index':
            if value <= 0.5:
                return '★★★★★'
            elif value <= 1.0:
                return '★★★★'
            elif value <= 1.5:
                return '★★★'
            elif value <= 2.0:
                return '★★'
            else:
                return '★'
        elif metric_name == 'Calinski-Harabasz Index':
            if value >= 1000:
                return '★★★★★'
            elif value >= 500:
                return '★★★★'
            elif value >= 100:
                return '★★★'
            elif value >= 10:
                return '★★'
            else:
                return '★'

    # Metrics dictionarie
    metrics_dict = {
        'Metric': [
            'Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index',
            'Homogeneity', 'Completeness', 'V-Measure', 'ARI', 'AMI'
        ],
        'Value': [
            silhouette, davies_bouldin, calinski_harabasz,
            homogeneity, completeness, v_measure, ari, ami
        ],
        'Rating': [
            rate_metric(silhouette, 'Silhouette Score'),
            rate_metric(davies_bouldin, 'Davies-Bouldin Index'),
            rate_metric(calinski_harabasz, 'Calinski-Harabasz Index'),
            rate_metric(homogeneity, 'Homogeneity'),
            rate_metric(completeness, 'Completeness'),
            rate_metric(v_measure, 'V-Measure'),
            rate_metric(ari, 'ARI'),
            rate_metric(ami, 'AMI')
        ]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    print(df_metrics)
