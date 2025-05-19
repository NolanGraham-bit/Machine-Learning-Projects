#************************************************************************************
# Nolan Graham
# ML – HW#2
# Filename: svm_model.py
# Due: Feb. 2, 2025
#
# Objective:
# To train and evaluate an SVM model on the WiFi channel classification dataset,
# perform PCA/LDA for feature selection, and generate evaluation metrics and visualizations.
#************************************************************************************

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import umap

# Step 1: Load the dataset
file_path = "WiFi_Channel_Labeled_Dataset.csv"

try:
    wifi_df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")
except Exception as e:
    print("Error loading file:", e)
    exit()

# Step 2: Extract features (X) and target labels (y)
X = wifi_df.drop(columns=['wifi_channel'])  # Features
y = wifi_df['wifi_channel']  # Target labels

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Perform PCA for feature selection
pca = PCA(n_components=2)  # Reduce to 2D for visualization
X_pca = pca.fit_transform(X_scaled)

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Train an SVM Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred_train = svm_model.predict(X_train)
y_pred_test = svm_model.predict(X_test)

# Step 8: Calculate Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Step 9: Save results to a text file
report_content = f"""
Machine Learning Model Evaluation Report
=======================================

Model: Support Vector Machine (SVM)

Training Accuracy: {train_accuracy:.4f}
Testing Accuracy: {test_accuracy:.4f}

Best Features Identified from PCA:
1. PC1 (Explains {pca.explained_variance_ratio_[0]:.2%} of variance)
2. PC2 (Explains {pca.explained_variance_ratio_[1]:.2%} of variance)

"""

with open("ML_Model_Evaluation_Report.txt", "w") as file:
    file.write(report_content)

print("SVM Model Evaluation Report saved!")

# Step 10: Generate UMAP Visualization
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="coolwarm", alpha=0.7)
plt.title("UMAP Visualization of Data (SVM Classification)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="WiFi Channel", loc="best")
plt.savefig("UMAP_Visualization_SVM.png")
plt.show()

# Step 11: Generate Decision Boundary Plot
fig, ax = plt.subplots(figsize=(8, 6))

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Train SVM on PCA-transformed data for visualization
svm_pca_model = SVC(kernel='linear', random_state=42)
svm_pca_model.fit(X_pca, y)

# Predict on grid
Z = svm_pca_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
ax.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, ax=ax, palette="coolwarm", alpha=0.7)
ax.set_title("Decision Boundary (SVM)")

plt.savefig("Decision_Boundary_SVM.png")
plt.show()
