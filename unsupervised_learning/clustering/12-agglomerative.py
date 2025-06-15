#!/usr/bin/env python3
"""Module for performing agglomerative clustering"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import numpy as np

def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2 or \
       not isinstance(dist, (int, float)) or dist <= 0:
        return None

    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    plt.figure(figsize=(10, 7))
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.title('Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')

    unique_clusters = np.unique(clss)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = dict(zip(unique_clusters, colors))

    leaf_colors = [cluster_colors[c] for c in 
                   clss[scipy.cluster.hierarchy.leaves_list(Z)]]
    scipy.cluster.hierarchy.set_link_color_palette(
        list(cluster_colors.values()))

    plt.show()

    return clss - 1



