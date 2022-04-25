import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

#print(len(selected_data))
def run_k_means_clustering(repaired_data):
    km = KMeans(
        n_clusters=2, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )

    y_km = km.fit_predict(X)
    # y_km= cluster
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.show()



def run_agglomerative_clustering(repaired_data):
    y_km = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    y_km= y_km.fit_predict(X)
    #y_km = km.fit_predict(X)
    # y_km= cluster
    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )
    plt.show()

def handle_missing_values(selected_data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(selected_data)
    # repaired_data = imp.transform(selected_data)

    return imp.transform(selected_data)




def get_output_column(repaired_data):
    output_column = []
    root_column = repaired_data.loc[:, 'root_depth_99']
    print(root_column.max())
    print(root_column.min())


data = pd.read_csv('data/camels_chars.txt', index_col='gauge_id')
ids = pd.read_csv('data/basin_list.txt', header=None, index_col=0)
selected_data = data[data.index.isin(ids.index)]

#run_k_means_clustering(repaired_data)
#run_agglomerative_clustering(repaired_data)

X=handle_missing_values(selected_data)



Y = get_output_column(selected_data)