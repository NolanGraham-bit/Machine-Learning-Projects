#************************************************************************************
# Nolan Graham
# ML – HW#3
# Filename: Hierarchical Analysis Combined.py
#
# Objective:
# To perform both K-Means and Agglomerative Clustering on the combined dataset (2011-2015),
# determine the optimal number of clusters using the Elbow Method, and generate a 
# hierarchical clustering dendrogram.
#************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import glob

# Load and combine datasets from 2011 to 2015
data_files = sorted(glob.glob("gt_*.csv"))  # Assumes files are named gt_2011.csv to gt_2015.csv
df_list = [pd.read_csv(file) for file in data_files]
df = pd.concat(df_list, ignore_index=True)

# Standardizing the dataset
X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig("elbow_method_combined.png")
plt.show()

# Step 2: Apply K-Means++ with optimal k
optimal_k = np.argmin(np.diff(inertia, 2)) + 2
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Save distortion score
with open("distortion_score_combined.txt", "w") as f:
    f.write(f"Optimal k: {optimal_k}\n")
    f.write(f"Distortion Score: {kmeans.inertia_}\n")

# Step 3: Plot the K-Means clusters
centers = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="viridis", alpha=0.6, edgecolor="k")
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label="Centroids")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.title(f'K-Means Clustering (k={optimal_k}) - Combined Dataset')
plt.legend()
plt.grid(True)
plt.savefig("kmeans_clusters_combined.png")
plt.show()

# Step 4: Perform silhouette analysis
silhouette_avg = silhouette_score(X_scaled, clusters)
sample_silhouette_values = silhouette_samples(X_scaled, clusters)

plt.figure(figsize=(8, 6))
y_lower = 10
for i in range(optimal_k):
    ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)

    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.title(f"Silhouette Analysis (k={optimal_k}) - Combined Dataset")
plt.savefig("silhouette_analysis_combined.png")
plt.show()

# Step 5: Perform Agglomerative Clustering and Generate Dendrogram
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram - Combined Dataset")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.savefig("dendrogram_combined.png")
plt.show()

print("All plots and results for the combined dataset have been saved successfully, including the dendrogram!")
