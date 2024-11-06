import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ucimlrepo import fetch_ucirepo 

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mutual_info_score, adjusted_rand_score 
from sklearn.preprocessing import  MinMaxScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import KernelPCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# fetch dataset 
wine = fetch_ucirepo(id=109) 
# data (as pandas dataframes) 

data = pd.DataFrame(wine.data.features)
y = pd.DataFrame(wine.data.targets) 

scaler = MinMaxScaler()

for col in data.columns:
    data[col] = scaler.fit_transform(data[[col]]).squeeze()


def elbow_curve(data):
    distorsions = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distorsions.append(kmeans.inertia_)
        
    plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()

log_columns = data.skew().sort_values(ascending=False)
log_columns = log_columns.loc[log_columns > 0.75]

data[log_columns.keys()] = data[log_columns.keys()].apply(np.log1p)

for col in data.columns:
    data[col] = scaler.fit_transform(data[[col]]).squeeze()

kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
X_kpca = kpca.fit_transform(data)

def clustering_metrics(X, target, labels):
    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    ch_index = calinski_harabasz_score(X, labels)
    ari = adjusted_rand_score(target, labels)
    mi = mutual_info_score(target, labels)

    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {db_index:.2f}")
    print(f"Calinski-Harabasz Index: {ch_index:.2f}")
    print(f"Adjusted Rand Index: {ari:.2f}")
    print(f"Mutual Information (MI): {mi:.2f}")

def kmeans_model(X):
    kmeans = KMeans(random_state=42, n_clusters=3, init='random')
    kmeans.fit(X)

    data['kmeans_3'] = kmeans.labels_
 
    return data['kmeans_3']
    
def gmm_model(X):
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)

    labels = gmm.predict(X)

    data['kmeans_3'] = labels
    
    return labels

def mean_shift_model(X):
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=len(X), random_state=42)
    
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
    
    labels = ms.labels_
    data['kmeans_3'] = labels
    
    return labels

def scatter_clusters(X, labels):
    plt.figure()
    scatter = sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='deep')
    scatter.legend_.set_title('Clusters')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
kmeans_labels = kmeans_model(X_kpca)
gmm_labels = gmm_model(X_kpca)
mean_shift_labels = mean_shift_model(X_kpca)

# print(mean_shift_labels)

clustering_metrics(X_kpca, y['class'], mean_shift_labels)
print()
# clustering_metrics(X_kpca, y['class'], gmm_labels)
# print()
# clustering_metrics(X_kpca, y['class'], mean_shift_labels)

scatter_clusters(X_kpca, mean_shift_labels)