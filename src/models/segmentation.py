import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_score


class Segmentation:
    def __init__(self):
        # self.data = data
        self.num_clusters = 2
        self.cluster_labels = None
        # self.data_norm = None

    # Column normaliztion
    def normalize(self, data, cols):
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data[cols])
        return pd.DataFrame(data_norm, columns=cols)

    def kmeans_segmentation(self, data, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        self.cluster_labels = kmeans.fit_predict(data)

    def elbow_method(self, data, max_clusters = 10):
        # Para trabajar con matrices y dataframes por igual
        data = np.array(data)
        nums_clusters = range(2, max_clusters+1)
        sums_of_squares = []
        for num_clusters in nums_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
            clusters = kmeans.fit_predict(data)
            # Se genera una matriz de la forma de data con el centroide asignado a cada punto
            centroides = kmeans.cluster_centers_[clusters]
            # La norma al cuadrado es la suma de los cuadrados
            sum_of_squares = ((centroides - data)**2).sum()
            sums_of_squares.append(sum_of_squares)
        return nums_clusters, sums_of_squares

    def gmm_segmentation(self, data, num_clusters):
        gmm = GaussianMixture(n_components=num_clusters, random_state=0)
        self.cluster_labels = gmm.fit_predict(data)

    def dbscan_segmentation(self, data, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(data)

    def jaccard_index(self, true_labels):
        intersection = np.sum((self.cluster_labels == 1) & (true_labels == 1))
        union = np.sum((self.cluster_labels == 1) | (true_labels == 1))
        return intersection / union

    def rand_index(self, true_labels):
        a = np.sum((self.cluster_labels == 1) & (true_labels == 1))
        b = np.sum((self.cluster_labels == 0) & (true_labels == 0))
        c = np.sum((self.cluster_labels == 1) & (true_labels == 0))
        d = np.sum((self.cluster_labels == 0) & (true_labels == 1))
        return (a + b) / (a + b + c + d)

    def stability_measures(self, X, labels, model):
        rand_scores = []
        jaccard_scores = []
        for _ in range(5):
            if model == 'kmeans':
                clustering_model = KMeans(n_clusters=self.num_clusters, random_state=np.random.randint(0, 100))
            elif model == 'gmm':
                clustering_model = GaussianMixture(n_components=self.num_clusters,
                                                   random_state=np.random.randint(0, 100))
            elif model == 'dbscan':
                clustering_model = GaussianMixture(n_components=self.num_clusters,
                                                   random_state=np.random.randint(0, 100))
            else:
                raise ValueError("Invalid model specified. Please use 'kmeans' or 'gmm'.")

            clustering_model.fit(X)
            new_labels = clustering_model.predict(X)

            rand_score = adjusted_rand_score(labels, new_labels)
            jaccard = jaccard_score(labels, new_labels, average="macro")

            rand_scores.append(rand_score)
            jaccard_scores.append(jaccard)

        return np.mean(rand_scores), np.mean(jaccard_scores)


def get_data_with_cluster(data, seg):
    data_copy = data.copy()
    data_copy['cluster'] = seg.cluster_labels
    return data_copy


def get_data_numeric_only(data_cluster):
    data_cluster_copy = data_cluster.copy()
    numeric_columns = list(data_cluster.select_dtypes(include=[int, float]).columns)
    return data_cluster_copy[numeric_columns]


def get_stats_table(data_cluster, col_name='cluster', stats_list=None):
    data_cluster_copy = data_cluster.copy()
    data_cluster_copy = get_data_numeric_only(data_cluster_copy)
    if stats_list is None:
        stats_list = ['mean', 'min', 'max', 'std']
    df_grouped = data_cluster_copy.groupby(col_name).agg(stats_list).reset_index()
    df_grouped.columns = list(map('_'.join, df_grouped.columns.values))
    return df_grouped
