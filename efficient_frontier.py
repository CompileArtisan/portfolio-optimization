import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp

# Load precomputed data
covariance_matrix = pd.read_csv('covariance_matrix.csv', index_col=0).values
expected_returns = pd.read_csv('expected_returns.csv', index_col=0).values.flatten()

# Define the number of assets
num_assets = len(expected_returns)

# Define range of target returns for the efficient frontier
target_returns = np.linspace(0.05, 0.25, 50)  # Modify range as needed
portfolio_risks = []
portfolio_weights = []

# Efficient Frontier Calculation
for target in target_returns:
    # Define variables
    weights = cp.Variable(num_assets)

    # Define constraints
    constraints = [
        cp.sum(weights) == 1,  # Weights must sum to 1
        weights >= 0          # No short selling
    ]
    constraints.append(weights @ expected_returns >= target)

    # Define the risk (variance) objective
    risk = cp.quad_form(weights, covariance_matrix)

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve()

    # Record the results
    portfolio_risks.append(np.sqrt(risk.value))  # Standard deviation = sqrt(variance)
    portfolio_weights.append(weights.value)

# Plot the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.plot(portfolio_risks, target_returns, label="Efficient Frontier")
plt.xlabel("Portfolio Risk (Standard Deviation)")
plt.ylabel("Portfolio Return")
plt.title("Efficient Frontier")
plt.legend()
plt.grid()
plt.show()

# Save the efficient frontier weights for further analysis
efficient_frontier_weights = pd.DataFrame(portfolio_weights, index=target_returns)
efficient_frontier_weights.to_csv('efficient_frontier_weights.csv')

print("Efficient frontier generated and weights saved.")
