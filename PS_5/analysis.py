import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Load the product and agent data
def load_data(data_path):
    product_data = pd.read_csv(data_path / 'data_market.csv')
    agent_data = pd.read_csv(data_path / 'data_agent.csv')
    return product_data, agent_data

# Prepare the data by calculating additional variables
def prepare_data(product_data):
    product_data['sum_shares'] = product_data.groupby('market_ids')['shares'].transform('sum')
    product_data['shares_0'] = 1 - product_data['sum_shares']
    product_data['log_sjt_s0t'] = np.log(product_data['shares'] / product_data['shares_0'])
    return product_data

# Define the function to estimate the linear model
def linear_model(params, x, p, log_sjt_s0t):
    beta0, beta1, beta2 = params
    predicted = beta0 + beta1 * x + beta2 * p
    residuals = log_sjt_s0t - predicted
    return np.sum(residuals**2)

# Fit the linear model
def fit_linear_model(product_data):
    initial_params = [0, 0, 0]
    result = minimize(linear_model, initial_params, args=(
        product_data['x1'], product_data['prices'], product_data['log_sjt_s0t']))
    return result.x

# Calculate additional variables for the approximation model
def calculate_additional_variables(product_data, agent_data):
    product_data['minvy_t'] = product_data['market_ids'].map(
        agent_data.groupby('market_ids')['income'].apply(lambda x: (1 / x).mean()))
    product_data['vinvy_t'] = product_data['market_ids'].map(
        agent_data.groupby('market_ids')['income'].apply(lambda x: (1 / x).var()))
    product_data['Kp_jt'] = (product_data['prices'] ** 2 - (product_data['shares'] * product_data['prices']).sum()) * product_data['prices']
    return product_data

# Function to estimate the approximation model
def approximation_model(params, x, p, log_sjt_s0t, minvy_t, vinvy_t, Kp_jt):
    beta0, beta1, alpha, alpha2 = params
    predicted = beta0 + beta1 * x + alpha * minvy_t * p + alpha2 * vinvy_t * Kp_jt
    residuals = log_sjt_s0t - predicted
    return np.sum(residuals**2)

# Fit the approximation model
def fit_approximation_model(product_data):
    initial_params = [0, 0, 0, 0]
    result = minimize(approximation_model, initial_params, args=(
        product_data['x1'], product_data['prices'], product_data['log_sjt_s0t'],
        product_data['minvy_t'], product_data['vinvy_t'], product_data['Kp_jt']
    ))
    return result.x

# Define the contraction mapping function
def contraction_mapping(delta, beta0, beta1, x, alpha, p, mu_t, sigma_t, share, num_simulations):
    y_it_samples = np.random.lognormal(mean=mu_t[:, np.newaxis], sigma=sigma_t[:, np.newaxis], size=(num_simulations, len(x)))
    utility = delta[:, np.newaxis] + beta0 + beta1 * x + (alpha * p / y_it_samples)
    exp_utility = np.exp(utility)
    est_share = exp_utility.mean(axis=0) / (1 + exp_utility.mean(axis=0))
    delta = delta + np.log(share) - np.log(est_share)
    return delta

# Iterative contraction mapping
def perform_contraction_mapping(product_data, beta0, beta1, alpha, num_simulations=1000, epsilon=1e-6):
    product_data['delta'] = np.zeros(len(product_data))
    mu_t = np.random.uniform(0.4, 0.8, num_simulations)
    sigma_t = np.random.uniform(0.2, 0.3, num_simulations)

    for _ in range(1000):
        delta_old = product_data['delta'].copy()
        product_data['delta'] = contraction_mapping(
            product_data['delta'], beta0, beta1, product_data['x1'], alpha,
            product_data['prices'], mu_t, sigma_t, product_data['shares'], num_simulations
        )
        if np.max(np.abs(product_data['delta'] - delta_old)) < epsilon:
            break
    return product_data['delta']

# Define the GMM objective function
def GMM_objective(params, product_data):
    beta0, beta1, alpha = params
    delta = np.zeros(len(product_data))
    max_iter = 10
    num_simulations = 1000
    epsilon = 1e-4

    mu_t = np.random.uniform(0.4, 0.8, num_simulations)
    sigma_t = np.random.uniform(0.2, 0.3, num_simulations)
    x1 = product_data['x1'].values
    prices = product_data['prices'].values

    for _ in range(max_iter):
        delta_old = delta.copy()
        y_it_samples = np.random.lognormal(mean=mu_t[:, np.newaxis], sigma=sigma_t[:, np.newaxis], size=(num_simulations, len(x1)))
        utility = delta[:, np.newaxis] + beta0 + beta1 * x1 + (alpha * prices / y_it_samples)
        exp_utility = np.exp(utility)
        est_share = exp_utility.mean(axis=1) / (1 + exp_utility.mean(axis=1))
        delta = delta + np.log(product_data['shares'].values) - np.log(est_share)
        if np.max(np.abs(delta - delta_old)) < epsilon:
            break

    residuals = delta - (beta0 + beta1 * x1 + alpha * prices / np.mean(y_it_samples, axis=0))
    w = product_data['w'].values
    G = (residuals[:, np.newaxis] * w).mean(axis=0)
    W = np.eye(len(G))
    Q = G.T @ W @ G
    return Q

# Fit the GMM model
def fit_GMM_model(product_data, initial_params):
    result = minimize(GMM_objective, initial_params, args=(product_data), method='BFGS')
    return result.x

# Main execution
data_path = 'your_data_path_here'  # Replace with your data path
product_data, agent_data = load_data(data_path)
product_data = prepare_data(product_data)
beta0, beta1, beta2 = fit_linear_model(product_data)
product_data = calculate_additional_variables(product_data, agent_data)
beta0, beta1, alpha, alpha2 = fit_approximation_model(product_data)
product_data['delta'] = perform_contraction_mapping(product_data, beta0, beta1, alpha)
initial_params = [beta0, beta1, alpha]
beta0, beta1, alpha = fit_GMM_model(product_data, initial_params)

print(f'Estimated parameters: beta0={beta0}, beta1={beta1}, alpha={alpha}')
