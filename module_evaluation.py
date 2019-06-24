from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from math import log2


def apply_kmeans_clustering(predictions, no_clusters):
    # Kmeans default parameters:
    # n_init=10 Number of time the Kmeans algorithm will be run with different centroid seeds
    kmeans = KMeans(n_clusters=no_clusters)

    # find the index of the cluster where each frames is assigned to
    prediction_clusterized = kmeans.fit_predict(predictions)

    return prediction_clusterized


def rearange_clustering_data(predictions_clusterized, no_clusters, ground_truths):
    # should be len(prediction_clusterized) == len(ground_truths), obv; that is the number of test frames
    if len(predictions_clusterized) != len(ground_truths):
        # again, really really hope this will never run
        raise Exception('get_tests_ground_truths() gave wrong output. len(predictions) != len(ground_truths)')

    # prepare the clustering data for computation of the entropy
    # matrix, row index is the cluster index; the row is the associated labels array with that cluster
    clusters_ground_truths = []
    for no_cluster in range(no_clusters):
        cluster_labels = []
        clustered_frames_indices = [index for index, value in enumerate(predictions_clusterized) if value == no_cluster]

        for index in clustered_frames_indices:
            cluster_labels.append(ground_truths[index])

        clusters_ground_truths.append(cluster_labels)

    return clusters_ground_truths


def calc_conditional_entropy(predictions_clusterized, no_clusters, ground_truths):
    clusters_ground_truths = rearange_clustering_data(predictions_clusterized, no_clusters, ground_truths)

    no_total_samples = len(predictions_clusterized)
    conditional_entropy = 0

    # calculate the conditional entropy
    for cluster_labels in clusters_ground_truths:
        # p(y)
        probl_cluster = len(cluster_labels) / no_total_samples

        distinct_cluster_labels = list(set(cluster_labels))
        no_samples_cluster = len(cluster_labels)

        # sum over X, labels associated to the cluster
        sum_over_ground_truths = 0
        for ground_truth in distinct_cluster_labels:
            label_count = cluster_labels.count(ground_truth)
            p_j = label_count / no_samples_cluster
            sum_over_ground_truths += p_j * log2(1 / p_j)

        conditional_entropy += probl_cluster * sum_over_ground_truths

    return conditional_entropy


def generate_2d_embedding(predictions):
    # the thing that created the 2-dimensional embedding for the 4096-dimensional features
    # TSNE default parameters: n_components=2 method='barnes_hut'
    prediction_tsne_embedded = TSNE().fit_transform(X=predictions)

    return prediction_tsne_embedded
