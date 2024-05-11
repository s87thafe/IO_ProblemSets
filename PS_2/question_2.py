import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from config import PS2

# Load your data
data_path = PS2.joinpath("data").resolve()
data = pd.read_csv(data_path/'WSDR.csv')

# Task 1: Descriptive statistics
# Rename columns to more descriptive names if necessary
data.rename(columns={'move': 'units_sold', 'custcoun': 'customer_count'}, inplace=True)

# Limit summary statistics to variables of interest:
data_subset = data[['units_sold', 'customer_count', 'price', 'profit']]

# Calculate descriptive statistics for the numerical columns
desc_stats = data_subset.describe().loc[['mean', 'std', 'min', 'max']]

# Display the descriptive statistics and the formatted DataFrame
print("Descriptive Statistics:")
print(desc_stats)


# Task 2: IV and OLS
# Calculating weekly market shares
data['market_shares'] = data.groupby(['week', 'store'])['units_sold'].transform(lambda x: x / x.sum())

# Calculate the logarithm of the weekly market share
data['log_market_share'] = np.log(data['market_shares'])

# Calculating the wholesale price
data['wholesale_price'] = data['price'] * (1 - data['profit'] / 100)

# Create dummy variables for the 'upc' column and join them back to the original dataset
data = pd.get_dummies(data, columns=['upc'], prefix='upc', drop_first=True)

# Calculate total units sold across all weeks
total_units_sold = data['units_sold'].sum()

# Subtract units sold per week from total units sold and subtract 1
data['market_size'] = total_units_sold / data.groupby('week')['units_sold'].transform('sum') - 1

# Calculate the logarithm of the market size
data['log_market_size'] = np.log(data['market_size'])

# Define the dependent variable
data['log_choice_prob'] = data['log_market_share'] - data['log_market_size']


# Run OLS
# Define the dependent variable
Y = data['log_choice_prob']

# Define the independent variables: include 'price' and all UPC dummies
X = data[['price'] + [col for col in data.columns if col.startswith('upc_')]]
X = sm.add_constant(X)  # adding a constant for the intercept

# Booleans are not supported by statsmodels, conver into integers
bool_cols = [col for col in X.columns if X[col].dtype == 'bool']
for col in bool_cols:
    X[col] = X[col].astype(int)

# Create the OLS model
model = sm.OLS(Y, X)

# Fit the model
results = model.fit()

# Print the results
print(results.summary())

# Run IV2SLS
instrument = data[['wholesale_price'] + [col for col in data.columns if col.startswith(('upc_'))]]
instrument = sm.add_constant(instrument)  # add a constant to the instruments

bool_cols = [col for col in instrument.columns if instrument[col].dtype == 'bool']
# Convert each boolean column to int
for col in bool_cols:
    instrument[col] = instrument[col].astype(int)

# Run the IV2SLS regression
iv_model = IV2SLS(Y, X, instrument).fit()

# Print the results
print(iv_model.summary())
