import pandas as pd

# Load the dataset
file_path_xlsx = "Data_User_Modeling_Dataset.xls"  # Ensure this file is in the same directory
xls = pd.ExcelFile(file_path_xlsx)

# Load relevant sheets
train_df = xls.parse("Training_Data").iloc[:, :6]  # Keep relevant columns
test_df = xls.parse("Test_Data").iloc[:, :6]  # Keep relevant columns

# Rename target column properly
train_df.rename(columns={" UNS": "UNS"}, inplace=True)
test_df.rename(columns={" UNS": "UNS"}, inplace=True)

# Standardize labels to lowercase and replace spaces with underscores
train_df["UNS"] = train_df["UNS"].str.lower().str.replace(" ", "_")
test_df["UNS"] = test_df["UNS"].str.lower().str.replace(" ", "_")
