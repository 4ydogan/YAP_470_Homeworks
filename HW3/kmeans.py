import random
import math

class KMeansClusterClassifier:
    def __init__(self, n_clusters, max_iterations=100):
        # Initialize the KMeansClusterClassifier with the desired number of clusters and max iterations
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = []

    def fit(self, X):
        # Fit the KMeans model to the input data X
        self.X = X
        self._initialize_centroids()  # Initialize centroids
        self._kmeans()  # Perform the KMeans clustering process

    def predict(self, X):
        # Predict the cluster assignments for the given data points in X
        predictions = [self._find_nearest_centroid(x) for x in X]
        return predictions

    def _initialize_centroids(self):
        # Initialize centroids by randomly selecting data points
        random_indices = random.sample(range(len(self.X)), self.n_clusters)
        self.centroids = [self.X[i] for i in random_indices]

    def _find_nearest_centroid(self, point):
        # Find the nearest centroid for a given data point
        distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
        return distances.index(min(distances))

    def _kmeans(self):
        # Perform the KMeans clustering process
        for _ in range(self.max_iterations):
            cluster_assignments = [self._find_nearest_centroid(x) for x in self.X]
            new_centroids = self._calculate_new_centroids(cluster_assignments)
            
            if self._has_converged(new_centroids):  # Check for convergence
                break

            self.centroids = new_centroids

    def _calculate_new_centroids(self, cluster_assignments):
        # Calculate new centroids based on cluster assignments
        new_centroids = []
        for cluster_id in range(self.n_clusters):
            cluster_points = [self.X[i] for i, c in enumerate(cluster_assignments) if c == cluster_id]
            if cluster_points:
                new_centroid = [sum(coord) / len(cluster_points) for coord in zip(*cluster_points)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(self.centroids[cluster_id])
        return new_centroids
    
    def _has_converged(self, new_centroids):
        # Check if the centroids have converged
        return all(self._arrays_equal(new, old) for new, old in zip(new_centroids, self.centroids))

    def _arrays_equal(self, arr1, arr2):
        # Check if two arrays are equal element-wise
        return all(x == y for x, y in zip(arr1, arr2))

    def _euclidean_distance(self, p1, p2):
        # Calculate the Euclidean distance between two data points p1 and p2
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))