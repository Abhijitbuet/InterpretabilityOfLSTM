import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
#print(len(selected_data))
def run_k_means_clustering(repaired_data):
    #repaired_data=repaired_data.reset_index()
    km = KMeans(
        n_clusters=4, init='random',
        n_init=10, max_iter=1000,
        tol=1e-04, random_state=0
    )
    repaired_data = handle_missing_values(repaired_data)
    repaired_data = normalize(repaired_data, axis=0)
    pca = PCA(n_components=2)
    repaired_data= pca.fit_transform(repaired_data)
    y_km = km.fit_predict(repaired_data)
    print(repaired_data[y_km == 0, 0])
    print(km.cluster_centers_)
    #return
    colors = ["red", "blue", "green", "orange"]
    for i in range (0,len(km.cluster_centers_)):
        plt.scatter(repaired_data[y_km == i, 0], repaired_data[y_km == i, 1],
        s=30, cmap='rainbow',
        marker='o', edgecolor='black',
        label='cluster '+str(i)
                 )

    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], label='Centers', c="black", s=100)
    distances = []
    for i in range(0, len(repaired_data)):
        distance =  numpy.linalg.norm(km.cluster_centers_[y_km[i]]- repaired_data[i])
        distances.append(distance)
    print(len(distances))
    df = pd.DataFrame(distances)

    df.sort_values(by= df.columns[0],inplace=True)
    print(df)
    outlier_start = len(df)*97.0/100
    i = 0
    outlier_data = []
    for index, row in df.iterrows():
        #print(index)
        if(i>outlier_start):
            outlier_data.append(repaired_data[index])
            y_km[index]=10
        i+=1
    print(len(outlier_data))
    print(repaired_data[y_km == 10, 1])
    plt.scatter(repaired_data[y_km == 10, 0], repaired_data[y_km == 10, 1], label='Centers', c="cyan", s=100)

    #print(df)
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
        s=50, c='yellow',
        marker='o', edgecolor='black',
        label='cluster 2'
    )
    plt.show()

def handle_missing_values(selected_data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(selected_data)
    # repaired_data = imp.transform(selected_data)

    return imp.transform(selected_data)




def select_k_for_kmeans(selected_data):
    selected_data = handle_missing_values(selected_data)
    Sum_of_squared_distances = []
    K = range(1, 15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(selected_data)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

data = pd.read_csv('data/camels_chars.txt', index_col='gauge_id')
ids = pd.read_csv('data/basin_list.txt', header=None, index_col=0)
selected_data = data[data.index.isin(ids.index)]
#select_k_for_kmeans(selected_data)
run_k_means_clustering(selected_data)
#run_agglomerative_clustering(repaired_data)

X=handle_missing_values(selected_data)



#Y = get_output_column(selected_data)