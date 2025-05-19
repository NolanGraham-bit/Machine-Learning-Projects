#************************************************************************************
# Nolan Graham
# ML – HW#2
# Filename: knn_model.py
# Due: Feb. 2, 2025
#
# Objective:
# Train and evaluate a k-NN model for WiFi channel classification,
# perform PCA for feature selection, and generate visualizations.
#************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import umap

# Load dataset
wifi_df = pd.read_csv("WiFi_Channel_Labeled_Dataset.csv")
X = wifi_df.drop(columns=['wifi_channel'])
y = wifi_df['wifi_channel']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for feature selection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Train k-NN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate Model
train_accuracy = accuracy_score(y_train, knn_model.predict(X_train))
test_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

# Append results to report
with open("ML_Model_Evaluation_Report.txt", "a") as file:
    file.write(f"\nModel: k-Nearest Neighbors\nTraining Accuracy: {train_accuracy:.4f}\nTesting Accuracy: {test_accuracy:.4f}\n")

# Generate Decision Boundary Plot
fig, ax = plt.subplots(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min()-1, X_pca[:, 0].max()+1, 200),
                     np.linspace(X_pca[:, 1].min()-1, X_pca[:, 1].max()+1, 200))
Z = knn_model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=ax, palette="coolwarm", alpha=0.7)
ax.set_title("Decision Boundary - k-NN")
plt.savefig("Decision_Boundary_kNN.png")
plt.show()
