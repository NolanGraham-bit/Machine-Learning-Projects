# ==========STEP 1: DATA LOADING, CLEANING, FEATURE ENGINEERING

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define dataset paths
plant_data_path = 'plant_1_generation.csv'  #Update these if needed
weather_data_path = 'plant_1_weather.csv'

####DATA LOADING

def load_data(plant_path, weather_path):
    try:
        plant_df = pd.read_csv(plant_path)
        weather_df = pd.read_csv(weather_path)
        print("Data loaded successfully!")
        return plant_df, weather_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

#######DATA CLEANING

def clean_data(plant_df, weather_df):
    #Check missing values
    print("\nMissing Values in Plant Data:\n", plant_df.isnull().sum())
    print("\nMissing Values in Weather Data:\n", weather_df.isnull().sum())

    #Drop duplicates
    plant_df.drop_duplicates(inplace=True)
    weather_df.drop_duplicates(inplace=True)

    #Parse timestamps
    plant_df['DATE_TIME'] = pd.to_datetime(plant_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    return plant_df, weather_df

###########FEATURE ENGINEERING

def feature_engineering(plant_df, weather_df):
    #Merge datasets on DATE_TIME
    merged_df = pd.merge_asof(
        plant_df.sort_values('DATE_TIME'),
        weather_df.sort_values('DATE_TIME'),
        on='DATE_TIME',
        direction='nearest'
    )

    #Create additional time features
    merged_df['DayOfYear'] = merged_df['DATE_TIME'].dt.dayofyear
    merged_df['WeekOfYear'] = merged_df['DATE_TIME'].dt.isocalendar().week
    merged_df['Hour'] = merged_df['DATE_TIME'].dt.hour
    merged_df['Weekday'] = merged_df['DATE_TIME'].dt.weekday
    merged_df['IsWeekend'] = merged_df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

    return merged_df

#########EDA Quick Check

def quick_eda(df):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("\nDataset Shape:", df.shape)
    print("\nDataset Columns:\n", df.columns)
    print("\nSample Data:\n", df.head())

    #Use only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()

    #Create friendly column name map
    rename_map = {
        'PLANT_ID_x': 'Plant ID X',
        'DC_POWER': 'DC Power',
        'AC_POWER': 'AC Power',
        'DAILY_YIELD': 'Daily Yield',
        'TOTAL_YIELD': 'Total Yield',
        'PLANT_ID_y': 'Plant ID Y',
        'AMBIENT_TEMPERATURE': 'Ambient Temp',
        'MODULE_TEMPERATURE': 'Module Temp',
        'IRRADIATION': 'Irradiation',
        'DayOfYear': 'Day of Year',
        'WeekOfYear': 'Week of Year',
        'Hour': 'Hour',
        'Weekday': 'Weekday',
        'IsWeekend': 'Weekend'
    }

    #Apply renaming
    correlation_matrix.rename(columns=rename_map, index=rename_map, inplace=True)

    #Plot
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        annot_kws={'size': 10},
        linewidths=0.5,
        cbar=True
    )

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout(pad=3.0)
    plt.savefig('heatmap_1_power_vs_irradiation.png')
    plt.show()



#========================MAIN========================#

if __name__ == "__main__":
    plant_df, weather_df = load_data(plant_data_path, weather_data_path)

    if plant_df is not None and weather_df is not None:
        plant_df, weather_df = clean_data(plant_df, weather_df)
        merged_df = feature_engineering(plant_df, weather_df)
        quick_eda(merged_df)

        # Save the processed dataset for modeling
        merged_df.to_csv('solar_data_cleaned.csv', index=False)
        print("\nProcessed data saved as 'solar_data_cleaned.csv'.")


