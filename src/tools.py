import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    background_image = plt.imread('../fondo.jpg')
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