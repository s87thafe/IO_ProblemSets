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
# Create a dictionary to hold the elasticity matrices for each (week, store)
elasticity_matrices = {}

# Extract unique weeks and sort them
unique_weeks = sorted(data['week'].unique())

# Extract unique stores and sort them
unique_stores = sorted(data['store'].unique())

# Group data by week and store
grouped = data.groupby(['week', 'store'])
for (week, store), group in grouped:
    n_products = len(group)

    # Capture prices and market shares from the current group
    prices = group['price'].values
    market_shares = group['market_shares'].values

    # Initialize the elasticity matrix
    elasticity_matrix = np.zeros((n_products, n_products))
    
    for i in range(n_products):
        for j in range(n_products):
            if i != j:
                elasticity_matrix[i, j] = -iv_model.params['price'] * prices[j] * market_shares[i]
            else:
                elasticity_matrix[i, j] = iv_model.params['price'] * prices[i] * (1 - market_shares[i])

    # Store the matrix
    elasticity_matrices[(week, store)] = elasticity_matrix

import numpy as np

# Task 5 and 6
# Calculate marginal costs for each (week, store)
marginal_costs_nb = {}
marginal_costs_jp = {}
all_prices = []
all_marginal_costs_nb = []
all_marginal_costs_jp = []

# Regularization value to avoid singular matrix issues
regularization_value = 1e-6

for key, elasticity_matrix in elasticity_matrices.items():
    week, store = key
    group = grouped.get_group((week, store))
    prices = group['price'].values
    market_shares = group['market_shares'].values

    # Initialize delta matrices
    delta_matrix_nb = np.zeros_like(elasticity_matrix)
    delta_matrix_jp = np.zeros_like(elasticity_matrix)
    
    for i in range(len(prices)):
        for j in range(len(prices)):
            delta_matrix_jp[i, j] = elasticity_matrix[i, j] * market_shares[i] / prices[j]
            if group['Brand'].iloc[i] == group['Brand'].iloc[j]:
                delta_matrix_nb[i, j] = elasticity_matrix[i, j] * market_shares[i] / prices[j]

    # Add regularization value to the diagonal
    delta_matrix_nb += np.eye(delta_matrix_nb.shape[0]) * regularization_value
    delta_matrix_jp += np.eye(delta_matrix_jp.shape[0]) * regularization_value

    # Calculate the marginal costs using the inverse of the delta matrix
    s = market_shares
    mc_nb = np.linalg.solve(delta_matrix_nb, s) + prices
    mc_jp = np.linalg.solve(delta_matrix_jp, s) + prices

    # Store the marginal costs
    marginal_costs_nb[key] = mc_nb
    marginal_costs_jp[key] = mc_jp
    
    # Collect all marginal costs and prices
    all_marginal_costs_nb.extend(mc_nb)
    all_marginal_costs_jp.extend(mc_jp)
    all_prices.extend(prices)

# Calculate the median marginal cost
median_mc_nb = np.median(all_marginal_costs_nb)
median_mc_jp = np.median(all_marginal_costs_jp)

# Calculate markups (p - mc) and relative markups (p - mc) / p
markups_nb = np.array(all_prices) - np.array(all_marginal_costs_nb)
relative_markups_nb = markups_nb / np.array(all_prices)
markups_jp = np.array(all_prices) - np.array(all_marginal_costs_jp)
relative_markups_jp = markups_jp / np.array(all_prices)

# Calculate median values for markups and relative markups
median_markup_nb = np.median(markups_nb)
median_relative_markup_nb = np.median(relative_markups_nb)
median_markup_jp = np.median(markups_jp)
median_relative_markup_jp = np.median(relative_markups_jp)

print("Median Marginal Cost Nash Bertrand:", median_mc_nb)
print("Median Markup Nash Bertrand (p - mc):", median_markup_nb)
print("Median Relative Markup Nash Bertrand (p - mc) / p:", median_relative_markup_nb)

print("Median Marginal Cost Joint Pricing:", median_mc_jp)
print("Median Markup Joint Pricing (p - mc):", median_markup_jp)
print("Median Relative Markup Joint Pricing (p - mc) / p:", median_relative_markup_jp)
