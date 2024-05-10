import pandas as pd
from config import PS2

# Load your data
data_path = PS2.joinpath("data").resolve()
data = pd.read_csv(data_path/'WSDR.csv')

# Task 1: Descriptive statistics
# Rename columns to more descriptive names if necessary
data.rename(columns={'move': 'units_sold', 'custcoun': 'customer_count'}, inplace=True)

# Calculate descriptive statistics for the numerical columns
desc_stats = data.describe().loc[['mean', 'std', 'min', 'max']]

# Display the descriptive statistics and the formatted DataFrame
print("Descriptive Statistics:")
print(desc_stats)