import pandas as pd
import numpy as np
from models.kmeans import KMeans  # Your KMeans implementation
from models.gmm import GMM  # Your GMM implementation
from models.pca import PCA  # Your PCA implementation
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_feather('data/external/word-embeddings.feather')

# Extract the embeddings (from the 'vit' column)
X = np.vstack(data['vit'])

# Step 1: Apply PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Step 2: Run K-Means clustering (use the optimal k from a4.py, assuming k=4)
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_pca)
kmeans_labels = kmeans.predict(X_pca)

# Step 3: Run GMM clustering (use the optimal k from a4.py, assuming k=4)
gmm = GMM(n_components=4)
gmm.fit(X_pca)
gmm_labels = gmm.predict(X_pca)

# Custom Silhouette Score Calculation
def silhouette_score(X, labels):
    n_samples = X.shape[0]
    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Points in the same cluster
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]
        
        # a(i): Average distance to other points in the same cluster
        a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1))
        
        # b(i): Average distance to points in the nearest other cluster
        b = np.inf
        for label in np.unique(labels):
            if label != labels[i]:
                b_cluster = np.mean(np.linalg.norm(X[i] - X[labels == label], axis=1))
                b = min(b, b_cluster)
        
        # Silhouette score for the sample
        silhouette_scores[i] = (b - a) / max(a, b)
    
    # Average Silhouette score
    return np.mean(silhouette_scores)

# Step 4: Evaluate Clustering Using Custom Silhouette Score
kmeans_silhouette = silhouette_score(X_pca, kmeans_labels)
gmm_silhouette = silhouette_score(X_pca, gmm_labels)

print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print(f"GMM Silhouette Score: {gmm_silhouette}")

# Step 5: Evaluate K-Means Clustering Using Inertia (WCSS)
kmeans_inertia = kmeans.getCost(X_pca)
print(f"K-Means Inertia (WCSS): {kmeans_inertia}")

# Step 6: Evaluate GMM Clustering Using Log-Likelihood
gmm_log_likelihood = gmm._compute_log_likelihood(X_pca)
print(f"GMM Log-Likelihood: {gmm_log_likelihood}")

# Step 7: Compare Results and Visualize Clusters

# K-Means Cluster Visualization
plt.figure(figsize=(8, 6))
scatter_kmeans = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', label='K-Means')
plt.legend(*scatter_kmeans.legend_elements(), title="K-Means Clusters")
plt.title("K-Means Clusters on PCA-Reduced Data")
plt.grid(True)
plt.savefig('figures/kmeans_pca_clusters_comparison.png')
print("K-Means clusters comparison plot saved as 'figures/kmeans_pca_clusters_comparison.png'")

# GMM Cluster Visualization
plt.figure(figsize=(8, 6))
scatter_gmm = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='plasma', label='GMM')
plt.legend(*scatter_gmm.legend_elements(), title="GMM Clusters")
plt.title("GMM Clusters on PCA-Reduced Data")
plt.grid(True)
plt.savefig('figures/gmm_pca_clusters_comparison.png')
print("GMM clusters comparison plot saved as 'figures/gmm_pca_clusters_comparison.png'")

# Step 8: Print Final Comparison
print("\n--- Final Comparison ---")
print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print(f"GMM Silhouette Score: {gmm_silhouette}")
print(f"K-Means Inertia (WCSS): {kmeans_inertia}")
print(f"GMM Log-Likelihood: {gmm_log_likelihood}")
