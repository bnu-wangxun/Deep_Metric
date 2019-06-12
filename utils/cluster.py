# coding=utf-8
from __future__ import absolute_import, print_function

import numpy as np

from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture


def cluster_(features,  labels, n_clusters):
    centers = []
    center_labels = []
    for label in set(labels):
        X = features[labels == label]
        kmeans = KMeans(n_clusters=n_clusters, random_state=None).fit(X)
        center_ = kmeans.cluster_centers_
        centers.extend(center_)
        center_labels.extend(n_clusters*[label])
    centers = np.conjugate(centers)
    centers = normalize(centers)
    return centers, center_labels


def normalize(X):
    norm_inverse = np.diag(1/np.sqrt(np.sum(np.power(X, 2), 1)))
    X_norm = np.matmul(norm_inverse, X)
    return X_norm





















