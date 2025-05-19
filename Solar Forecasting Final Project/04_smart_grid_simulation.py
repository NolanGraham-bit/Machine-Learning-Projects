#=================SMART GRID SIMULATION DEMO LOGIC=================#

###This is too simulate basic storage behavior based on predicted solar yield####

import pandas as pd
import matplotlib.pyplot as plt

#####Load predictions from XGBoost
#####Simulate storage logic on X_test + predictions
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

#Load data
data = pd.read_csv("solar_data_cleaned.csv")

#####Define features/target for simulation
feature_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                'DayOfYear', 'WeekOfYear', 'Hour', 'Weekday', 'IsWeekend']
target_col = 'DAILY_YIELD'

X = data[feature_cols]
y = data[target_col]

#####Chronological split
split_index = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

#Train XGBoost model
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Simulate basic storage logic:
#== - If predicted > 250, store energy (battery charging)
#== - If predicted < 100, discharge battery to compensate
#== - Otherwise, no action (grid balanced)

simulation_df = X_test.copy()
simulation_df["Predicted_Yield"] = y_pred
simulation_df["Battery_Action"] = simulation_df["Predicted_Yield"].apply(
    lambda y: "Charge" if y > 250 else ("Discharge" if y < 100 else "Stable")
)

#Count of actions
action_counts = simulation_df["Battery_Action"].value_counts()

#Plot action distribution
plt.figure(figsize=(6,4))
action_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Smart Grid Battery Actions Based on Predicted Yield")
plt.xlabel("Action")
plt.ylabel("Number of Intervals")
plt.tight_layout()
plt.savefig("smart_grid_simulation_actions.png")
plt.show()

#Print action summary
print("\nBattery Action Summary:")
print(action_counts)

#Save simulated output
simulation_df.to_csv("Smart_Grid_Simulation_Output.csv", index=False)
print("\nSimulation output saved as 'Smart_Grid_Simulation_Output.csv'")


