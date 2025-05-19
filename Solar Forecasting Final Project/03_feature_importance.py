#========================STEP 3: FEATURE IMPORTANCE PLOTS========================#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

#Load dataset
data = pd.read_csv("solar_data_cleaned.csv")

# Define input features and target
feature_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                'DayOfYear', 'WeekOfYear', 'Hour', 'Weekday', 'IsWeekend']
target_col = 'DAILY_YIELD'

X = data[feature_cols]
y = data[target_col]

#Fit Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

#Fit XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X, y)

#Create feature importance plot function
def plot_feature_importance(model, feature_names, title, save_name):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()

#Plot and save
plot_feature_importance(rf_model, feature_cols, 'Random Forest Feature Importance', 'rf_feature_importance.png')
plot_feature_importance(xgb_model, feature_cols, 'XGBoost Feature Importance', 'xgb_feature_importance.png')

print("\nFeature importance plots saved as 'rf_feature_importance.png' and 'xgb_feature_importance.png'.")

