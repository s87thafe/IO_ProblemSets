import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
from config import PS2

# Load data
data_path = PS2.joinpath("data").resolve()
data = pd.read_csv(data_path/'WSDR.csv')

# Task 1: Descriptive statistics

# Rename columns
data.rename(columns={'move': 'units_sold', 'custcoun': 'customer_count'}, inplace=True)

# Limit summary statistics to variables of interest:
data_subset = data[['units_sold', 'customer_count', 'price', 'profit']]

# Calculate descriptive statistics for the numerical columns
desc_stats = data_subset.describe().loc[['mean', 'std', 'min', 'max']]

# Limit counting unique values to categorial variables
categorial_vars = data[['store', 'upc', 'Brand', 'week']]

# Create a dictionary to store the unique counts
unique_counts = {col: len(data[col].unique()) for col in categorial_vars}

# Convert the dictionary to a DataFrame
unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Variable', 'Unique Values'])

# Display the DataFrame
print("Unique Counts:")
print(unique_counts_df)

# Display the descriptive statistics and the formatted DataFrame
print("Descriptive Statistics:")
print(desc_stats)

# Task 2: IV and OLS

# Calculating market shares
data['market_shares'] = data['units_sold']/data['customer_count']
data['log_market_share'] = np.log(data['market_shares'])

# Calculate total units sold across all weeks
unique_pairs = data.groupby(['week', 'store'])['customer_count'].first()
total_customers = unique_pairs.sum()

# Correct iteration over MultiIndex
for key in unique_pairs.index:
    week, store = key
    # Perform operations with week and store

# Sum of sold units per market
data['total_units_sold'] = data.groupby(['week', 'store'])['units_sold'].transform('sum')

# Benchmark market size
data['market_size'] = (total_customers - data['total_units_sold']) / data['customer_count']
data['log_market_size'] = np.log(data['market_size'])
cd 
# Define the dependent variable
data['log_choice_prob'] = data['log_market_share'] - data['log_market_size']

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

# Run OLS
# Define the variables
X_OLS = data[['price'] + [col for col in data.columns if col.startswith('upc_')]]
X_OLS = sm.add_constant(X_OLS)  # adding a constant for the intercept
Y = data['log_choice_prob']
OLS_model = sm.OLS(Y, X_OLS)
results = OLS_model.fit()
print(results.summary())

# Run IV2SLS
instrument = data[['wholesale_price'] + [col for col in data.columns if col.startswith(('upc_'))]]
instrument = sm.add_constant(instrument)
IV_model = IV2SLS(Y, X_OLS, instrument).fit()
print(IV_model.summary())

# Task 3
# Discuss the relevance of the IV
X_OLS = sm.add_constant(data['wholesale_price'])

# Run the first-stage regression
first_stage = sm.OLS(data['price'], X_OLS).fit()

# Print the summary of the regression to see the coefficient and p-value
print(first_stage.summary())

# Task 4 and 6
# Calculate own-price and cross-price elasticities and Marginal Costs
# Create Unique Market IDs
data['unique_id'] = data['store'].astype(str) + '_' + data['week'].astype(str)
data['unique_market_id'] = pd.factorize(data['unique_id'])[0] + 1
unique_markets = data['unique_market_id'].unique()

elasticity_matrices = {}
own_price_elasticities_by_product = {}
cross_price_elasticities_by_product = {}

# Initialize columns for marginal costs
data['marginal_cost_NB'] = np.nan
data['marginal_cost_JP'] = np.nan

for market in unique_markets:
    # Filter data for the current market
    market_data = data[data['unique_market_id'] == market]
    
    # Determine the number of products
    n_products = market_data['price'].shape[0]  # Assuming 'price' column has one price per product
    
    # Extract prices and market shares for the current market
    prices = market_data['price'].values
    market_shares = market_data['market_shares'].values
    
    # Initialize the elasticity matrix
    elasticity_matrix = np.zeros((n_products, n_products))
    delta_matrice_NB_init = np.zeros((n_products, n_products))
    delta_matrice_JP_init = np.zeros((n_products, n_products))

    for i in range(n_products):
        product_i = market_data['original_upc'].iloc[i]
        if product_i not in own_price_elasticities_by_product:
            own_price_elasticities_by_product[product_i] = []
        if product_i not in cross_price_elasticities_by_product:
            cross_price_elasticities_by_product[product_i] = []
        for j in range(n_products):
            if i != j:
                # Cross-price elasticity
                elasticity_matrix[i, j] = -IV_model.params['price'] * prices[j] * market_shares[i]
                cross_price_elasticities_by_product[product_i].append(elasticity_matrix[i, j])
                if market_data['Brand'].iloc[i] == market_data['Brand'].iloc[j]:
                    delta_matrice_NB_init[i, j] = elasticity_matrix[i, j] * (market_shares[i] / prices[i])
                delta_matrice_JP_init[i, j] = elasticity_matrix[i, j] * (market_shares[i] / prices[i])
            else:
                # Own-price elasticity
                elasticity_matrix[i, j] = IV_model.params['price'] * prices[i] * (1 - market_shares[i])
                own_price_elasticities_by_product[product_i].append(elasticity_matrix[i, j])
                if market_data['Brand'].iloc[i] == market_data['Brand'].iloc[j]:
                    delta_matrice_NB_init[i, j] = elasticity_matrix[i, j] * (market_shares[i] / prices[i])
                delta_matrice_JP_init[i, j] = elasticity_matrix[i, j] * (market_shares[i] / prices[i])
    
    delta_inv_NB = np.linalg.inv(delta_matrice_NB_init)
    delta_inv_JP = np.linalg.inv(delta_matrice_JP_init)
    
    # Calculate marginal costs for each product
    marginal_costs = np.dot(delta_inv_NB, market_shares) + prices
    
    # Update the main DataFrame with calculated marginal costs
    data.loc[data['unique_market_id'] == market, 'marginal_cost'] = marginal_costs
    
    # Store the elasticity matrix for the current market
    elasticity_matrices[market] = elasticity_matrix

# Calculate the median elasticity by product
own_price_medians_by_product = {product: np.median(elasticities) for product, elasticities in own_price_elasticities_by_product.items()}
cross_price_medians_by_product = {product: np.median(elasticities) for product, elasticities in cross_price_elasticities_by_product.items()}

# Output the results
print("Median Own Price Elasticities by Product:")
print(own_price_medians_by_product)

print("\nMedian Cross Price Elasticities by Product:")
print(cross_price_medians_by_product)

print("\nCalculated Marginal Cost:")
print(data['marginal_cost'].median())

# Task 5
# Single Product Nash Bertrand: Assume that all firms except firm i keep their prices constant
# Convergence parameters
threshold = 1e-6
max_iterations = 1000

for market in unique_markets:
    # Filter data for the current market
    market_data = data[data['unique_market_id'] == market]
    
    # Determine the number of products
    n_products = market_data.shape[0]

    # Initialize the elasticity matrix
    delta_matrice_SNB = np.zeros((n_products, n_products))
    elasticity_matrix_SNB = np.zeros((n_products, n_products))

    # Initialize prices
    market_data['predicted_prices_SNB'] = market_data['price'].values
    x = 0

    while x < max_iterations:
        x += 1
        previous_prices = market_data['predicted_prices_SNB'].copy()

        # Extract prices and market shares for the current market
        selected_columns = ['predicted_prices_SNB'] + [col for col in market_data.columns if col.startswith('upc_')]
        algo_X = market_data[selected_columns]
        algo_X = sm.add_constant(algo_X)
        market_data['predicted_market_shares'] = np.exp(IV_model.predict(algo_X))
        market_data['predicted_market_shares'] /= 1 + market_data['predicted_market_shares'].sum()
        
        market_shares = market_data['predicted_market_shares'].values
        prices = market_data['predicted_prices_SNB'].values
        
        delta_inv_SNB = np.linalg.inv(delta_matrice_NB_init)
        market_data['predicted_prices_NB'] = market_data['marginal_cost'] - np.dot(delta_inv_SNB, market_shares)

        # Check for convergence
        if np.all(np.abs(previous_prices - market_data['predicted_prices_SNB']) < threshold):
            break

    # Store the final prices in the main DataFrame
    data.loc[data['unique_market_id'] == market, 'predicted_markup_SNB'] = market_data['predicted_prices_SNB']-market_data['marginal_cost']
    data.loc[data['unique_market_id'] == market, 'predicted_relative_markup_SNB'] = data['predicted_markup_SNB'] / market_data['predicted_prices_SNB']
    
# Nash Bertrand equilibrium
# Convergence parameters
threshold = 1e-6
max_iterations = 1000

for market in unique_markets:
    # Filter data for the current market
    market_data = data[data['unique_market_id'] == market]
    
    # Determine the number of products
    n_products = market_data.shape[0]

    # Initialize the elasticity matrix
    delta_matrice_NB = np.zeros((n_products, n_products))
    elasticity_matrix_NB = np.zeros((n_products, n_products))

    # Initialize prices
    market_data['predicted_prices_NB'] = market_data['price'].values
    x = 0

    while x < max_iterations:
        x += 1
        previous_prices = market_data['predicted_prices_NB'].copy()

        # Extract prices and market shares for the current market
        selected_columns = ['predicted_prices_NB'] + [col for col in market_data.columns if col.startswith('upc_')]
        algo_X = market_data[selected_columns]
        algo_X = sm.add_constant(algo_X)
        market_data['predicted_market_shares'] = np.exp(IV_model.predict(algo_X))
        market_data['predicted_market_shares'] /= 1 + market_data['predicted_market_shares'].sum()
        market_shares = market_data['predicted_market_shares'].values
        prices = market_data['predicted_prices_NB'].values

        for i in range(n_products):
            for j in range(n_products):
                if i != j:
                    # Cross-price elasticity
                    elasticity_matrix_NB[i, j] = -IV_model.params['price'] * prices[j] * market_shares[i]
                    if market_data['Brand'].iloc[i] == market_data['Brand'].iloc[j]:
                        delta_matrice_NB[i, j] = elasticity_matrix_NB[i, j] * (market_shares[i] / prices[i])
                else:
                    # Own-price elasticity
                    elasticity_matrix_NB[i, j] = IV_model.params['price'] * prices[i] * (1 - market_shares[i])
                    if market_data['Brand'].iloc[i] == market_data['Brand'].iloc[j]:
                        delta_matrice_NB[i, j] = elasticity_matrix_NB[i, j] * (market_shares[i] / prices[i])
        
        delta_inv_NB = np.linalg.inv(delta_matrice_NB)
        market_data['predicted_prices_NB'] = market_data['marginal_cost'] - np.dot(delta_inv_NB, market_shares)

        # Check for convergence
        if np.all(np.abs(previous_prices - market_data['predicted_prices_NB']) < threshold):
            break

    # Store the final prices in the main DataFrame
    data.loc[data['unique_market_id'] == market, 'predicted_markup_NB'] = market_data['predicted_prices_NB']-market_data['marginal_cost']
    data.loc[data['unique_market_id'] == market, 'predicted_relative_markup_NB'] = data['predicted_markup_NB'] / market_data['predicted_prices_NB']


# Convergence parameters
threshold = 1e-3
max_iterations = 1000
determinanten = []

for market in unique_markets:
    # Filter data for the current market
    market_data = data[data['unique_market_id'] == market]
    
    # Determine the number of products
    n_products = market_data.shape[0]

    # Initialize the elasticity matrix
    delta_matrice_JP = np.zeros((n_products, n_products))
    elasticity_matrix_JP = np.zeros((n_products, n_products))

    # Initialize prices
    market_data['predicted_prices_JP'] = market_data['price'].values
    x = 0

    while x < max_iterations:
        x += 1
        previous_prices = market_data['predicted_prices_JP'].copy()
        
        # Extract prices and market shares for the current market
        selected_columns = ['predicted_prices_JP'] + [col for col in market_data.columns if col.startswith('upc_')]
        algo_X = market_data[selected_columns]
        algo_X = sm.add_constant(algo_X)
        
        # Predict market shares
        market_data['predicted_market_shares'] = np.exp(IV_model.predict(algo_X))
        market_data['predicted_market_shares'] /= 1 + market_data['predicted_market_shares'].sum()
        market_shares = market_data['predicted_market_shares'].values
        prices = market_data['predicted_prices_JP'].values
        
        for i in range(n_products):
            for j in range(n_products):
                if i != j:
                    # Cross-price elasticity
                    elasticity_matrix_JP[i, j] = -IV_model.params['price'] * prices[j] * market_shares[i]
                    delta_matrice_JP[i, j] = elasticity_matrix_JP[i, j] * (market_shares[i] / prices[i])
                else:
                    # Own-price elasticity
                    elasticity_matrix_JP[i, j] = IV_model.params['price'] * prices[i] * (1 - market_shares[i])
                    delta_matrice_JP[i, j] = elasticity_matrix_JP[i, j] * (market_shares[i] / prices[i])

        # To avoid numerical issues, we normalize the matrix            
        scaling_factor = np.linalg.norm(delta_matrice_JP, ord=np.inf)
        normalized_matrix = delta_matrice_JP / scaling_factor

        # Calculate the determinant of the matrix
        inverse_normalized_matrix = np.linalg.inv(normalized_matrix)
        
        # Denormalize again
        delta_inv_JP = inverse_normalized_matrix / scaling_factor
        
        market_data['predicted_prices_JP'] = market_data['marginal_cost'] - np.dot(delta_inv_JP, market_shares)
        
        # Check for convergence
        if np.all(np.abs(previous_prices - market_data['predicted_prices_JP']) < threshold):
            break

    # Store the final prices in the main DataFrame
    data.loc[data['unique_market_id'] == market, 'predicted_markup_JP'] = market_data['predicted_prices_JP']-market_data['marginal_cost']
    data.loc[data['unique_market_id'] == market, 'predicted_relative_markup_JP'] = data['predicted_markup_JP'] / market_data['predicted_prices_JP']
    
print("\nCalculated Single Product Nash Bertrand Markups:")
print(data['predicted_markup_SNB'].describe())
print("\nCalculated Single Product Nash Bertrand Median Markups:")
print(data['predicted_markup_SNB'].median())
print("\nCalculated Single Product Nash Bertrand Median Relative Markups:")
print(data['predicted_relative_markup_SNB'].median())

print("\nCalculated Nash Bertrand Median Markups:")
print(data['predicted_markup_NB'].describe())
print("\nCalculated Nash Bertrand Median Markups:")
print(data['predicted_markup_NB'].median())
print("\nCalculated Nash Bertrand Median Relative Markups:")
print(data['predicted_relative_markup_NB'].median())

print("\nCalculated Joint Pricing Markups:")
print(data['predicted_markup_JP'].describe())
print("\nCalculated Joint Pricing Median Markups:")
print(data['predicted_markup_JP'].median())
print("\nCalculated Joint Pricing Median Relative Markups:")
print(data['predicted_relative_markup_JP'].median())
