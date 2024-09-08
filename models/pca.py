import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        # Step 1: Mean center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(X_centered.T)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 4: Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

        # Step 5: Select the first n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project the data onto the principal components
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
