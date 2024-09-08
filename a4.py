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

# Step 2: Apply K-Means clustering with Elbow Method to find optimal k
def elbow_method(X, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(kmeans.getCost(X))
    
    # Plot the WCSS for each k
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k (K-Means)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.savefig('figures/kmeans_elbow_method.png')
    print("Elbow plot saved as 'figures/kmeans_elbow_method.png'")
    return wcss

# Run the Elbow Method for K-Means
wcss = elbow_method(X_pca, max_k=10)

# You can manually select the optimal k based on the plot (for demonstration, assume k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(X_pca)
kmeans_labels = kmeans.predict(X_pca)

# Step 3: Apply GMM clustering with BIC to find optimal number of components
def gmm_bic(X, max_k=10):
    bic_values = []
    for k in range(1, max_k + 1):
        gmm = GMM(n_components=k)
        gmm.fit(X)
        bic = gmm.getBIC(X)
        bic_values.append(bic)
    
    # Plot the BIC for each k
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), bic_values, marker='o', linestyle='--')
    plt.title('BIC for Optimal k (GMM)')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('BIC Score')
    plt.grid(True)
    plt.savefig('figures/gmm_bic_method.png')
    print("BIC plot saved as 'figures/gmm_bic_method.png'")
    return bic_values

# Run the BIC Method for GMM
bic_values = gmm_bic(X_pca, max_k=10)

# Again, for demonstration purposes, let's assume GMM with k=4
gmm = GMM(n_components=4)
gmm.fit(X_pca)
gmm_labels = gmm.predict(X_pca)

# Step 4: Visualize both K-Means and GMM clusters
# K-Means Cluster Visualization
plt.figure(figsize=(8, 6))
scatter_kmeans = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', label='K-Means')
plt.legend(*scatter_kmeans.legend_elements(), title="K-Means Clusters")
plt.title("K-Means Clusters on PCA-Reduced Data")
plt.grid(True)
plt.savefig('figures/kmeans_pca_clusters.png')
print("K-Means clusters plot saved as 'figures/kmeans_pca_clusters.png'")

# GMM Cluster Visualization
plt.figure(figsize=(8, 6))
scatter_gmm = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='plasma', label='GMM')
plt.legend(*scatter_gmm.legend_elements(), title="GMM Clusters")
plt.title("GMM Clusters on PCA-Reduced Data")
plt.grid(True)
plt.savefig('figures/gmm_pca_clusters.png')
print("GMM clusters plot saved as 'figures/gmm_pca_clusters.png'")
