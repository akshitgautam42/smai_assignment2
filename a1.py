import pandas as pd
import numpy as np
from models.kmeans import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_feather('data/external/word-embeddings.feather')

# Extract the embeddings (from the 'vit' column)
X = np.vstack(data['vit'])

# Function to apply the Elbow Method
def plot_elbow_method(X, max_k=10):
    wcss = []
    
    # Loop over different values of k
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(kmeans.getCost(X))
    
    # Plot the WCSS for each k
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.savefig('figures/elbow_method.png')
    print("Elbow plot saved to 'figures/elbow_method.png'")

# Apply the Elbow Method
plot_elbow_method(X, max_k=25)

# Apply KMeans clustering with the optimal number of clusters 
optimal_k = 4 
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(X)

# Get the cluster labels and cost
labels = kmeans.predict(X)
cost = kmeans.getCost(X)

print(f"Cluster labels with k={optimal_k}:", labels)
print(f"K-Means Cost (WCSS) with k={optimal_k}:", cost)
