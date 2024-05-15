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

own_price_elasticities = []
cross_price_elasticities = []

for matrix in elasticity_matrices.values():
    # Extract diagonal (own-price elasticities)
    own_price_elasticities.extend(np.diag(matrix))
    
    # Extract off-diagonal (cross-price elasticities)
    n = matrix.shape[0]  # Size of the matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                cross_price_elasticities.append(matrix[i, j])

# Calculate median of own-price and cross-price elasticities
median_own_price = np.median(own_price_elasticities)
median_cross_price = np.median(cross_price_elasticities)

print("Median Own-Price Elasticity:", median_own_price)
print("Median Cross-Price Elasticity:", median_cross_price)

# Task 5
# Nash Bertrand equilibrium
# Calculate marginal costs for each (week, store)
marginal_costs = {}
all_marginal_costs = []
all_prices = []

for key, elasticity_matrix in elasticity_matrices.items():
    week, store = key
    group = grouped.get_group((week, store))
    prices = group['price'].values
    market_shares = group['market_shares'].values

    # Derive the delta matrix
    delta_matrix = np.zeros_like(elasticity_matrix)
    for i in range(len(prices)):
        for j in range(len(prices)):
            if group['Brand'].iloc[i] == group['Brand'].iloc[j]:
                delta_matrix[i, j] = elasticity_matrix[i, j] * market_shares[i] / prices[j]

    try:
        delta_matrix_inv = np.linalg.inv(delta_matrix)
    except np.linalg.LinAlgError:
        delta_matrix_inv = np.linalg.pinv(delta_matrix)

    # Calculate the marginal costs
    s = market_shares
    mc = np.dot(delta_matrix_inv, s) + prices

    # Store the marginal costs
    marginal_costs[key] = mc
    
    # Collect all marginal costs and prices
    all_marginal_costs.extend(mc)
    all_prices.extend(prices)

# Calculate the median marginal cost
median_mc = np.median(all_marginal_costs)

# Calculate markups (p - mc) and relative markups (p - mc) / p
markups = np.array(all_prices) - np.array(all_marginal_costs)
relative_markups = markups / np.array(all_prices)

# Calculate median values for markups and relative markups
median_markup = np.median(markups)
median_relative_markup = np.median(relative_markups)

print("Median Marginal Cost:", median_mc)
print("Median Markup (p - mc) Nash Equilibrium:", median_markup)
print("Median Relative Markup (p - mc) / p Nash Equilibrium:", median_relative_markup)

# Joint Pricing:
# Best-response mapping function without regularization
def best_response_mapping(prices, mc, elasticity_matrix, market_shares):
    n_products = len(prices)
    delta_matrix = np.zeros_like(elasticity_matrix)
    
    for i in range(n_products):
        for j in range(n_products):
            delta_matrix[i, j] = elasticity_matrix[i, j] * market_shares[i] / prices[j]
    
    q = market_shares
    eta = np.linalg.lstsq(delta_matrix**(-1), q, rcond=None)[0]
    return mc + eta

# Normalize prices and market shares
def normalize(values):
    mean_value = np.mean(values)
    if mean_value == 0:
        raise ValueError("Mean of the values is zero, cannot normalize.")
    return values / mean_value

# Set a convergence threshold
threshold = 1e-2
max_iterations = 100

# Initial prices normalization
initial_prices = {key: normalize(group['price'].values) for key, group in grouped}

# Iterate over each (week, store) pair
joint_prices = {}
iteration_counts = {}
for key, elasticity_matrix in elasticity_matrices.items():
    week, store = key
    group = grouped.get_group((week, store))
    prices = initial_prices[key]
    market_shares = group['market_shares'].values
    mc = marginal_costs[key]

    # Initialize the price vector
    p_current = prices
    iteration_count = 0
    for iteration in range(max_iterations):
        iteration_count += 1
        # Calculate the new prices using the best-response mapping
        p_new = best_response_mapping(p_current, mc, elasticity_matrix, market_shares)
        
        # Check for convergence
        if np.max(np.abs(p_new - p_current)) < threshold:
            break
        p_current = p_new

    # Store the converged prices and iteration count
    joint_prices[key] = p_current
    iteration_counts[key] = iteration_count

# Collect all joint prices for further analysis
all_joint_prices = [price for prices in joint_prices.values() for price in prices]

# Diagnostic: Print median prices and marginal costs before calculating markups
print("Median Prices:", np.median(all_joint_prices))
print("Median Marginal Costs:", np.median(all_marginal_costs))

# Calculate markups (p - mc) and relative markups (p - mc) / p for joint prices
joint_markups = np.array(all_joint_prices) - np.array(all_marginal_costs)
joint_relative_markups = joint_markups / np.array(all_joint_prices)

# Calculate median values for joint markups and relative markups
median_joint_markup = np.median(joint_markups)
median_joint_relative_markup = np.median(joint_relative_markups)

print("Median Joint Markup (p - mc):", median_joint_markup)
print("Median Joint Relative Markup (p - mc) / p:", median_joint_relative_markup)