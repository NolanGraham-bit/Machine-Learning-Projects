import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

#Load dataset
df = pd.read_csv("solar_data_cleaned.csv")

#Sort chronologically
df = df.sort_values('DATE_TIME').reset_index(drop=True)

#Create lag features for next 3 days (96 steps per day for 15-min intervals)
df['DAILY_YIELD_T+1'] = df['DAILY_YIELD'].shift(-96)
df['DAILY_YIELD_T+2'] = df['DAILY_YIELD'].shift(-96 * 2)
df['DAILY_YIELD_T+3'] = df['DAILY_YIELD'].shift(-96 * 3)

#Drop missing values
df_clean = df.dropna(subset=['DAILY_YIELD_T+1', 'DAILY_YIELD_T+2', 'DAILY_YIELD_T+3'])

#Feature columns
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
            'DayOfYear', 'WeekOfYear', 'Hour', 'Weekday', 'IsWeekend']

X = df_clean[features]
y1 = df_clean['DAILY_YIELD_T+1']
y2 = df_clean['DAILY_YIELD_T+2']
y3 = df_clean['DAILY_YIELD_T+3']

#Train & predict function with reduced complexity for speed
def train_predict_fast(X, y):
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    model = XGBRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test.reset_index(drop=True), pd.Series(y_pred)

#Run predictions
y1_true, y1_pred = train_predict_fast(X, y1)
y2_true, y2_pred = train_predict_fast(X, y2)
y3_true, y3_pred = train_predict_fast(X, y3)

#lot results
plt.figure(figsize=(10, 6))
plt.plot(y1_true[:200], label='Actual Day+1', alpha=0.7)
plt.plot(y1_pred[:200], label='Predicted Day+1', linestyle='--')

plt.plot(y2_true[:200], label='Actual Day+2', alpha=0.7)
plt.plot(y2_pred[:200], label='Predicted Day+2', linestyle='--')

plt.plot(y3_true[:200], label='Actual Day+3', alpha=0.7)
plt.plot(y3_pred[:200], label='Predicted Day+3', linestyle='--')

plt.title('Multistep Forecasting: Actual vs Predicted Yield (3-Day Horizon)')
plt.xlabel('Time Steps')
plt.ylabel('Daily Yield (kWh)')
plt.legend()
plt.tight_layout()
plt.savefig("multistep_forecast_plot.png")
plt.show()
