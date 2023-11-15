import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np


def read_data(data_path):
    credit_card_data = pd.read_csv(data_path)
    data = np.array(credit_card_data)
    np.random.shuffle(data)
    return data


def split_data(data):
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    return X_train, X_test, y_train, y_test


def filter_clusters(X_train, y_train, df_clusters):
    df_clusters['Class'] = y_train
    class_mask = df_clusters['Class'] == 1
    class_fitered_df = df_clusters[class_mask]
    relevant_clusters = class_fitered_df['Clusters'].value_counts().index.values[0]  # select the top two
    
    print("Selecting following clusters which has most positive classes: ", relevant_clusters)
    
    X_train['Clusters'] = df_clusters['Clusters']
    mask = X_train['Clusters'] == relevant_clusters
    X_train_clustered = X_train[mask]
    y_train_clustered = y_train[mask]

    return X_train_clustered, y_train_clustered


def DBSCAN_Clustering(data_raw, features_of_interest, epsilon, min_samp, flag):
    # if flag:
        # from sklearnex import patch_sklearn  # pylint: disable=E0401, C0415
        # patch_sklearn()
    from sklearn.cluster import DBSCAN  # pylint: disable=C0415
    scaler = StandardScaler()
    data_for_clustering = data_raw[features_of_interest]
    data_for_clustering = pd.DataFrame(data_for_clustering)
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)
    lst_clustering_time = []
    for _i in [1, 2, 3, 4, 5]:
        start_time = time.time()
        db = DBSCAN(eps=epsilon, min_samples=min_samp, n_jobs=-1).fit(data_for_clustering_scaled)
        lst_clustering_time.append(time.time()-start_time)
    clustering_time = min(lst_clustering_time)
    data_for_clustering['Clusters'] = db.labels_
    return data_for_clustering, clustering_time


def cluster(X_train, y_train, X_test, y_test):
    most_important_names = 16
    eps_val = 0.3
    minimum_samples = 20
    df_clusters, cluster_time = DBSCAN_Clustering(
        X_train, most_important_names, eps_val, minimum_samples, flag="train"
    )
    X_train_clustered, y_train_clustered = filter_clusters(
        X_train, y_train, df_clusters
    )

    X_test['Class'] = y_test
    X_test.to_csv("./data/creditcard_test.csv", index=False)

    print("DBSCAN Clustering time: ", cluster_time)
    return X_train_clustered, y_train_clustered
