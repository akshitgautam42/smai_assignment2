import pandas as pd
import numpy as np
from models.gmm import GMM  # Import the custom GMM class
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_feather('data/external/word-embeddings.feather')

# Extract the embeddings (from the 'vit' column)
X = np.vstack(data['vit'])

# Test different values of k for GMM and calculate AIC/BIC
aic_values = []
bic_values = []
n_components_range = range(1, 10)  # Test from 1 to 9 components

for k in n_components_range:
    gmm = GMM(n_components=k)
    gmm.fit(X)
    aic = gmm.getAIC(X)
    bic = gmm.getBIC(X)
    aic_values.append(aic)
    bic_values.append(bic)
    print(f"GMM with k={k}: AIC={aic}, BIC={bic}")

# Plot AIC and BIC to determine the optimal k
plt.figure(figsize=(8, 6))
plt.plot(n_components_range, aic_values, label='AIC', marker='o')
plt.plot(n_components_range, bic_values, label='BIC', marker='o')
plt.title('AIC and BIC for Different k (Number of Components)')
plt.xlabel('Number of Components (k)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('figures/aic_bic_plot.png')
print("AIC/BIC plot saved as 'aic_bic_plot.png'")


optimal_k = np.argmin(bic_values) + 1  # Add 1 since range starts at 1
print(f"Optimal number of components (k) based on BIC: {optimal_k}")

# Apply GMM clustering with the optimal number of components
gmm = GMM(n_components=optimal_k)
gmm.fit(X)
gmm_labels = gmm.predict(X)
print(f"GMM Cluster labels with k={optimal_k}: {gmm_labels}")
