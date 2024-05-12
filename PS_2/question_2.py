import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from config import PS2

# Load your data
data_path = PS2.joinpath("data").resolve()
data = pd.read_csv(data_path/'WSDR.csv')

# Task 1: Descriptive statistics
# Rename columns
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

# Preserve original data
data['original_upc'] = data['upc']

# Create dummy variables for the 'upc' column
upc_dummies = pd.get_dummies(data['upc'], prefix='upc', drop_first=True)

# Join the dummy variables back to the original dataset
data = pd.concat([data.drop('upc', axis=1), upc_dummies], axis=1)

# Ensure all dummy columns are of integer type
for col in upc_dummies.columns:
    data[col] = data[col].astype(int)

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

# Create the OLS model
model = sm.OLS(Y, X)

# Fit the model
results = model.fit()

# Print the results
print(results.summary())

# Run IV2SLS
instrument = data[['wholesale_price'] + [col for col in data.columns if col.startswith(('upc_'))]]
instrument = sm.add_constant(instrument)  # add a constant to the instruments

# Run the IV2SLS regression
iv_model = IV2SLS(Y, X, instrument).fit()

# Extract the coefficient for 'price'
beta_price = iv_model.params['price']

# Print the results
print(iv_model.summary())

# Task 3
# Discuss the relevance of the IV
X = sm.add_constant(data['wholesale_price'])

# Run the first-stage regression
first_stage = sm.OLS(data['price'], X).fit()

# Print the summary of the regression to see the coefficient and p-value
print(first_stage.summary())

# Testing Exogeneity
# Calculate residuals from the first stage
data['residuals'] = first_stage.resid

# Add a constant to the regression model
X_resid = sm.add_constant(data['residuals'])

# Regress the dependent variable on the residuals
check_exogeneity = sm.OLS(data['log_choice_prob'], X_resid).fit()

# Print the results
print(check_exogeneity.summary())

# Task 4
# Compute the quantity sold for each product in each market
data['quantity'] = data['market_shares'] * data.groupby(['week', 'store'])['units_sold'].transform('sum')

# Calculate own-price elasticity for each product in each market
data['own_price_elasticity'] = iv_model.params['price'] * data['price'] * (1 - data['market_shares'])

# Calculate cross-price elasticities

data['cross_price_elasticity'] = -iv_model.params['price'] * data['price'] * data['market_shares']


# Median of own-price elasticity across markets
median_own_price_elasticity = data.groupby(['week', 'store'])['own_price_elasticity'].median()
median_cross_price_elasticity = data.groupby(['week', 'store'])['cross_price_elasticity'].median()
print("Median Own-Price Elasticity across markets:", median_own_price_elasticity)
print("Median Cross-Price Elasticity across markets:", median_cross_price_elasticity)
