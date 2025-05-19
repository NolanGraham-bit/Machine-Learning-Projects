#========================STEP 2: MODEL TRAINING & EVALUATION========================#


###Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#########LOAD PROCESSED DATA

def load_processed_data(filepath):
    df = pd.read_csv(filepath)
    return df

#########DATA PREPARATION

def prepare_data(df):
    #Define input features and target
    feature_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                    'DayOfYear', 'WeekOfYear', 'Hour', 'Weekday', 'IsWeekend']
    target_col = 'DAILY_YIELD'

    X = df[feature_cols]
    y = df[target_col]

    #Chronological 80/20 split
    split_index = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

########Model Training

def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models

#############Model Evaluation

def evaluate_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R^2': r2
        })

        #Plot true vs predicted
        plt.figure(figsize=(8,5))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'{name} - Actual vs Predicted Daily Yield')
        plt.xlabel('Sample')
        plt.ylabel('Daily Yield (kWh)')
        plt.legend()
        plt.show()

    return pd.DataFrame(results)

#====================================MAIN========================#

if __name__ == "__main__":
    data = load_processed_data('solar_data_cleaned.csv')
    X_train, X_test, y_train, y_test = prepare_data(data)

    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)

    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))

    #Save results for report
    results_df.to_csv('Model_Performance_Summary.csv', index=False)
    print("\nPerformance summary saved as 'Model_Performance_Summary.csv'.")

