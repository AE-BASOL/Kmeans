import numpy as np
from random import sample

class KMeansManual:
    def __init__(self, n_clusters=5, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = []

    def initialize_centroids(self, data):
        self.centroids = data[sample(range(len(data)), self.n_clusters)]

    def assign_clusters(self, data):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, data_point in enumerate(data):
            distances = [np.linalg.norm(data_point - centroid) for centroid in self.centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(idx)
        return clusters

    def calculate_new_centroids(self, data, clusters):
        new_centroids = []
        for cluster in clusters:
            new_centroids.append(np.mean([data[idx] for idx in cluster], axis=0))
        return new_centroids

    def fit(self, data):
        self.initialize_centroids(data)
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(data)
            new_centroids = self.calculate_new_centroids(data, clusters)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return clusters

    def repeat_kmeans(self, data, true_labels, num_repeats=10):
        accuracy_list = []
        for _ in range(num_repeats):
            self.fit(data)
            predicted_clusters = self.assign_clusters(data)
            accuracy = self.calculate_accuracy(predicted_clusters, true_labels)
            accuracy_list.append(accuracy)
        return accuracy_list

    def calculate_accuracy(self, predicted_clusters, true_labels):
        correct_count = 0
        for cluster in predicted_clusters:
            label_counts = {label: 0 for label in set(true_labels)}
            for idx in cluster:
                label_counts[true_labels[idx]] += 1
            most_common_label = max(label_counts, key=label_counts.get)
            correct_count += label_counts[most_common_label]
        total_points = len(true_labels)
        return correct_count / total_points if total_points > 0 else 0
