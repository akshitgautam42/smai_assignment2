import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def initialize_centroids(self, X):
        np.random.seed(42)
        random_indices = np.random.permutation(X.shape[0])
        centroids = X[random_indices[:self.n_clusters]]
        return centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_iter):
            self.labels = self.assign_clusters(X, self.centroids)
            new_centroids = self.compute_centroids(X, self.labels)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def compute_centroids(self, X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def predict(self, X):
        return self.assign_clusters(X, self.centroids)

    def getCost(self, X):
        distances = np.sqrt(((X - self.centroids[self.labels]) ** 2).sum(axis=1))
        return np.sum(distances ** 2)
