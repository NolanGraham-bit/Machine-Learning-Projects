###************************************************************************************
# Nolan Graham
# ML – HW#4
# Filename: ml_hw4.ipynb
# Due: 3/6/2025
#
# Objective:
# To preprocess data, train machine learning models, evaluate performance, and visualize results.
###************************************************************************************

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Read the dataset file and convert to a DataFrame
file_path = "Concrete_Data.xlsx"  
df = pd.read_excel(file_path)

# Rename columns for better readability
df.columns = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", 
              "Coarse Aggregate", "Fine Aggregate", "Age", "Strength"]

# Display the first five rows
print("First five rows of the dataset:")
print(df.head())

# Step 3: Generate the correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Step 4: Separate feature columns from the target column
X = df.drop(columns=['Strength'])  # Features
y = df['Strength']  # Target variable

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Apply Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Step 7: Apply Decision Tree Regressor
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train_scaled, y_train)
y_pred_dtr = dtr.predict(X_test_scaled)

# Step 8: Apply Support Vector Machine Regressor
svr = SVR()
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)

# Step 9: Evaluate models
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mse_dtr = mean_squared_error(y_test, y_pred_dtr)
r2_dtr = r2_score(y_test, y_pred_dtr)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Print evaluation results
print("\nModel Evaluation Results:")
print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression R²: {r2_lr}")
print(f"Decision Tree Regressor MSE: {mse_dtr}")
print(f"Decision Tree R²: {r2_dtr}")
print(f"SVR MSE: {mse_svr}")
print(f"SVR R²: {r2_svr}")

# Step 10: Create bar plots for comparison
plt.figure(figsize=(12,5))

# MSE Plot
plt.subplot(1, 2, 1)
plt.bar(['Linear Regression', 'Decision Tree', 'SVR'], [mse_lr, mse_dtr, mse_svr], color=['blue', 'orange', 'green'])
plt.xlabel("Models")
plt.ylabel("MSE")
plt.title("MSE Comparison")

# RSquared Score Plot
plt.subplot(1, 2, 2)
plt.bar(['Linear Regression', 'Decision Tree', 'SVR'], [r2_lr, r2_dtr, r2_svr], color=['blue', 'orange', 'green'])
plt.xlabel("Models")
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.tight_layout()
plt.show()

# Step 11: Plot regression lines
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_dtr, color='orange', label='Decision Tree')
plt.scatter(y_test, y_pred_svr, color='green', label='SVR')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='black', linewidth=2)
plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("Actual vs. Predicted Concrete Strength for Different Models")
plt.legend()
plt.show()
