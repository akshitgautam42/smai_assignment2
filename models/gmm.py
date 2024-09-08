import numpy as np

class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.responsibilities_ = np.zeros((n_samples, self.n_components))

    def _e_step(self, X):
        log_resp = np.zeros_like(self.responsibilities_)

        for k in range(self.n_components):
            log_resp[:, k] = np.log(self.weights_[k] + 1e-10) + self._log_multivariate_gaussian(X, self.means_[k], self.covariances_[k])

        # Normalize the responsibilities in log-space
        log_sum_resp = np.logaddexp.reduce(log_resp, axis=1)
        self.responsibilities_ = np.exp(log_resp - log_sum_resp[:, np.newaxis])

    def _m_step(self, X):
        N_k = self.responsibilities_.sum(axis=0)
        for k in range(self.n_components):
            self.means_[k] = (1 / N_k[k]) * np.dot(self.responsibilities_[:, k], X)
            centered_X = X - self.means_[k]
            self.covariances_[k] = (1 / N_k[k]) * np.dot(self.responsibilities_[:, k] * centered_X.T, centered_X)
            self.weights_[k] = N_k[k] / X.shape[0]

    def _compute_log_likelihood(self, X, epsilon=1e-10):
        log_likelihood = 0
        for k in range(self.n_components):
            # Add epsilon to prevent issues with very small log values
            gaussian_log_prob = self._log_multivariate_gaussian(X, self.means_[k], self.covariances_[k], epsilon)
            
            # Use the max value between the computed log and epsilon to prevent issues
            log_likelihood += self.weights_[k] * np.maximum(gaussian_log_prob, np.log(epsilon))
    
        if np.isnan(log_likelihood).any() or np.isinf(log_likelihood).any():
            return -np.inf  # Set to a very negative value if instability occurs
        return np.log(log_likelihood + epsilon).sum()



    def _log_multivariate_gaussian(self, X, mean, covariance, epsilon=1e-6):
        n_features = X.shape[1]
        diff = X - mean
        
        # Add small value to diagonal for numerical stability
        covariance += np.eye(n_features) * epsilon

        L = np.linalg.cholesky(covariance)
        log_det_cov = 2 * np.sum(np.log(np.diag(L)))

        inv_cov = np.linalg.inv(L).T @ np.linalg.inv(L)
        log_exp_term = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

        return log_exp_term - 0.5 * (n_features * np.log(2 * np.pi) + log_det_cov)


    def fit(self, X):
        self._initialize_parameters(X)
        log_likelihoods = []
        for i in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            if i > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

    def predict(self, X):
        self._e_step(X)
        return np.argmax(self.responsibilities_, axis=1)
    
    
    def getAIC(self, X):
        log_likelihood = self._compute_log_likelihood(X)
        if np.isinf(log_likelihood) or np.isnan(log_likelihood):
            return np.inf  # Return a very large number if log-likelihood is invalid
        num_params = self.n_components * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2)  # mean + cov params
        return 2 * num_params - 2 * log_likelihood

    def getBIC(self, X):
        log_likelihood = self._compute_log_likelihood(X)
        if np.isinf(log_likelihood) or np.isnan(log_likelihood):
            return np.inf  # Return a very large number if log-likelihood is invalid
        num_params = self.n_components * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2)  # mean + cov params
        n = X.shape[0]
        return np.log(n) * num_params - 2 * log_likelihood

