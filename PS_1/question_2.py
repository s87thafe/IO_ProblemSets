import numpy as np
from scipy.optimize import minimize
import pandas as pd


def log_likelihood(params, data):
    alpha_kinoko, alpha_takenoko, beta = params
    income = 300  # Fixed income as per your problem statement
    log_likelihood = 0
    
    for _, row in data.iterrows():
        p_kinoko = np.exp(alpha_kinoko + beta * (income - row['pKinoko']))
        p_takenoko = np.exp(alpha_takenoko + beta * (income - row['pTakenoko']))
        p_other = 1  # Base probability for the outside option
        
        sum_probs = p_kinoko + p_takenoko + p_other
        p_kinoko /= sum_probs
        p_takenoko /= sum_probs
        p_other /= sum_probs
        
        # Assign the correct probability based on the observed choice
        if row['choice'] == 1:
            log_likelihood += np.log(p_kinoko)
        elif row['choice'] == 2:
            log_likelihood += np.log(p_takenoko)
        else:
            log_likelihood += np.log(p_other)
    
    return -log_likelihood  # Negative for maximization via minimization function

# Load data
data = pd.read_csv('data_KinokoTakenoko.csv')

# Initial parameter estimates
initial_params = np.array([0.0, 0.0, 0.0])

result = minimize(log_likelihood, initial_params, args=(data,), method='BFGS')

if result.success:
    print('Optimization was successful.')
    print('Estimated parameters:', result.x)
else:
    print('Optimization failed.')
    print(result.message)
