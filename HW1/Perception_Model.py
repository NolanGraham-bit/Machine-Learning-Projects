#************************************************************************************
# Nolan Graham
# ML – HW#1
# Filename: perceptron_model.py
# Due: Feb. 25, 2025
#
# Objective:
# Implement a Perceptron model to classify labels using the STG and PEG features.
#*************************************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "Data_User_Modeling_Dataset.xls"  # Ensure this file is in the same directory
xls = pd.ExcelFile(file_path)
train_df = xls.parse("Training_Data").iloc[:, :6]
test_df = xls.parse("Test_Data").iloc[:, :6]

# Rename target column
train_df.rename(columns={" UNS": "UNS"}, inplace=True)
test_df.rename(columns={" UNS": "UNS"}, inplace=True)

# Encode target labels
train_df["UNS"] = train_df["UNS"].str.lower().str.replace(" ", "_")
test_df["UNS"] = test_df["UNS"].str.lower().str.replace(" ", "_")

label_mapping = {label: idx for idx, label in enumerate(train_df["UNS"].unique())}
train_df["UNS"] = train_df["UNS"].map(label_mapping)
test_df["UNS"] = test_df["UNS"].map(label_mapping)

# Feature selection
X_train = train_df[["STG", "PEG"]].values
y_train = train_df["UNS"].values
X_test = test_df[["STG", "PEG"]].values
y_test = test_df["UNS"].values

# Standardizing features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train Perceptron
ppn = Perceptron(max_iter=100, eta0=0.01, random_state=1, shuffle=True)
ppn.fit(X_train_std, y_train)

# Predict and evaluate
y_pred = ppn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy results
with open("perceptron_accuracy.txt", "w") as f:
    f.write(f"Perceptron Test Accuracy: {accuracy:.4f}\n")

# Plot classification results
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='o', label='True Labels')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='x', label='Predicted Labels')
plt.xlabel("STG")
plt.ylabel("PEG")
plt.title("Perceptron Classification Results")
plt.legend()
plt.savefig("perceptron_classification.png")
plt.show()
