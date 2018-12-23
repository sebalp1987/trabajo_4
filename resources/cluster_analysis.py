from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
import matplotlib.pyplot as plot
import matplotlib.cm as cm
from itertools import cycle
from random import randint
from sklearn.cluster import DBSCAN
import pandas as pd
import STRING
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import safe_indexing, check_X_y
from sklearn.metrics import pairwise_distances

def find_min_distances(x1, x2):
    min_dist = np.zeros(len(x1))

    for i, x1_i in enumerate(x1):
        dists = np.sqrt(np.sum((x2 - x1_i) ** 2, axis=1))
        min_dist[i] = dists.min()

    return min_dist


def split_data_random(x, split_size):
    # copy data
    xc = np.copy(x)

    # split
    lo = randint(1, len(x) - split_size) - 1
    up = lo + split_size
    xsp = xc[lo:up:1]

    # remaining
    xre = np.delete(xc, np.s_[lo:up:1], 0)
    return xsp, xre


def expl_hopkins(x, split_size=50, num_iters=10):
    seed = 0
    np.random.seed(seed)
    n, d = x.shape  # obtenemos int de la dimensión de x

    xr = np.random.random((n, d))

    print("calculating hopkins stats to detect if the data set has clusters...")

    hopkins_stats = []

    for i in range(0, num_iters):
        (X_spl, X_tra) = split_data_random(x, split_size)
        (X_ran, X_rem) = split_data_random(xr, split_size)

        min_dist_ran = find_min_distances(X_ran, X_tra)
        min_dist_spl = find_min_distances(X_spl, X_tra)

        # print("random")
        # print min_dist_ran
        ran_sum = min_dist_ran.sum()
        # print("sum %.3f" % (ran_sum))

        # print("split")
        # print min_dist_spl
        spl_sum = min_dist_spl.sum()
        # print("sum %.3f" % (spl_sum))

        hopkins_stat = spl_sum / (ran_sum + spl_sum)
        print("hopkins stats %.3f" % hopkins_stat)
        hopkins_stats.append(hopkins_stat)

    av_hopkins_stat = np.mean(hopkins_stats)
    print("average hopkins stat %.3f" % av_hopkins_stat)


def hopkins(X):
    d = X.shape[1]
    # d = len(vars) # columns
    n = len(X)  # rows
    m = int(0.1 * n)  # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2,
                                    return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(
            ujd, wjd)
        H = 0

    return H


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.
    Parameters
    ----------
    n_labels : int
        Number of labels
    n_samples : int
        Number of samples
    """
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)


def davies_bouldin_score(X, labels):
    """Computes the Davies-Bouldin score.
    The score is defined as the ratio of within-cluster distances to
    between-cluster distances.
    Read more in the :ref:`User Guide <davies-bouldin_index>`.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)
    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(
            cluster_k, [centroid]))

    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    score = (intra_dists[:, None] + intra_dists) / centroid_distances
    score[score == np.inf] = np.nan
    return np.mean(np.nanmax(score, axis=1))


def cluster_internal_validation(x, n_clusters, model=None):
    lscores = []

    for nc in range(2, n_clusters + 1):
        print(nc)
        if model is None:
            km = KMeans(n_clusters=nc, random_state=10, init='k-means++', n_init=100, max_iter=300)
        else:
            km = DBSCAN(eps=0.5, min_samples=10, leaf_size=30, n_jobs=-1)

        labels = km.fit_predict(x)
        lscores.append((
                        silhouette_score(x, labels),
                        calinski_harabaz_score(x, labels),
                        davies_bouldin_score(x, labels)

                        ))

    print(lscores)
    fig = plot.figure(figsize=(15, 5))
    fig.add_subplot(131)
    plot.plot(range(2, n_clusters + 1), [x for x, _, _ in lscores])
    plot.title('Silhoutte Score')
    plot.xlabel('Number of Clusters')
    fig.add_subplot(132)
    plot.plot(range(2, n_clusters + 1), [x for _, x, _ in lscores])
    plot.title('Calinski-Harabaz Score')
    plot.xlabel('Number of Clusters')
    fig.add_subplot(133)
    plot.plot(range(2, n_clusters + 1), [x for _, _, x in lscores])
    plot.title('Davies-Bouldin Index')
    plot.xlabel('Number of Clusters')
    plot.show()


def silhouette_coef(x, range_n_clusters, model=None):
    for n_clusters in range_n_clusters:

        # Create a subplot wtih 1 row and 2 cols
        fig, (ax1, ax2) = plot.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Primer Subplot: Silhouette plot con rango x [-1,1] y rango y [0, len(X)]
        # pero le agregamos un espacio en blanco (n_clusters+1)*10
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

        # modelo de cluster
        if model is None:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        else:
            clusterer = DBSCAN(eps=range_n_clusters, min_samples=200, leaf_size=30, n_jobs=-1)
        cluster_labels = clusterer.fit_predict(x)

        # Silhoutte score average for all samples
        silhouette_avg = silhouette_score(x, cluster_labels)
        print('For n_clusters = ', n_clusters, ' The average SC is: ', silhouette_avg)

        # Silhoutte score for each sample
        sample_silhouette_values = silhouette_samples(x, cluster_labels)

        y_lower = 10

        # Agregando los SC para los samples por cluster i y ordenarlos
        for i in range(n_clusters):
            ith_cluster_sc_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_sc_values.sort()

            size_cluster_i = ith_cluster_sc_values.shape[0]
            y_upper = y_lower + size_cluster_i
            cmap = cm.get_cmap('Spectral')
            color = cmap(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_sc_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Le ponemos el nombre del cluster en el grÃ¡fico
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2do Plot: Mostramos los verdaderos clusters formados
        cmap = cm.get_cmap('Spectral')
        colors = cmap(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(x[:, 0], x[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Ponemos nombre al cluster
        centers = clusterer.cluster_centers_

        # Dibujamos el centro con un cÃ­ruclo
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plot.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                       "with n_clusters = %d" % n_clusters),
                      fontsize=14, fontweight='bold')

        plot.show()


def kmeans_plus_plus(x, k, n_init, max_iter, show_plot=True, drop='total_code', file_name='clusters'):
    """
    :param show_plot:
    :param drop:
    :return:
    :param x:
    :param k: number_clusters
    :param n_init: Number of time the k-means algorithm will be run with different centroid seeds
    :param max_iter:Maximum number of iterations of the k-means algorithm for a single run
    :return:
    """
    if drop is not None:
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=0).fit(
            x.drop(drop, axis=1))
    else:
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=0).fit(
            x)

    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    df = pd.DataFrame(labels, columns=['labels'], index=x.index)
    df = pd.concat([x, df], axis=1)
    df = df.copy()
    df.to_csv(STRING.path_db_aux + '\\' + file_name + '.csv', sep=';', index=True, encoding='latin1')
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    if drop is not None:
        x = x.drop(drop, axis=1).values
    else:
        x = x.values
    if show_plot:
        plot.figure(1)
        plot.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plot.plot(x[my_members, 0], x[my_members, 1], col + '.')
            plot.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=14)
        plot.title('Estimated number of clusters: %d' % n_clusters_)
        plot.show()

    return df
