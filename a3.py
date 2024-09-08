import pandas as pd
import numpy as np
from models.gmm import GMM  # Your custom GMM implementation
from models.pca import PCA  # Custom PCA implementation
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_feather('data/external/word-embeddings.feather')

# Extract the embeddings (from the 'vit' column)
X = np.vstack(data['vit'])

# Step 1: Apply PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Step 2: Apply GMM clustering on the PCA-reduced data
gmm = GMM(n_components=4)
gmm.fit(X_pca)
gmm_labels = gmm.predict(X_pca)

# Step 3: Plot the clusters after PCA reduction
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis')

# Add a legend
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.title("GMM Clusters Visualized with Custom PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

# Step 4: Save the plot as an image
plt.savefig('figures/gmm_pca_clusters.png')
print("Plot saved as 'gmm_pca_clusters.png'")
