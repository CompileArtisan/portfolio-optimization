import cvxpy as cp
import numpy as np
import pandas as pd

# Load precomputed data
covariance_matrix = pd.read_csv('covariance_matrix.csv', index_col=0).values
expected_returns = pd.read_csv('expected_returns.csv', index_col=0).values.flatten()

# Define the number of assets
num_assets = len(expected_returns)

# Define variables
weights = cp.Variable(num_assets)

# Define constraints
constraints = [
    cp.sum(weights) == 1,  # Weights must sum to 1
    weights >= 0          # No short selling
]

# Define the risk (variance) objective
risk = cp.quad_form(weights, covariance_matrix)

# Define the target return (modify this as needed)
target_return = 0.1
constraints.append(weights @ expected_returns >= target_return)

# Solve the optimization problem
problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()

# Results
optimal_weights = weights.value
print("Optimal Weights:", optimal_weights)
# print(sum(optimal_weights))  # Should be close to 1
