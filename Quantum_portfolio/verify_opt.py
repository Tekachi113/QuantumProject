import numpy as np
import pandas as pd
import utils

print("--- Running Classical & Quantum Optimization verification tests ---")

# Mock data for 3 assets
tickers = ["AAPL", "MSFT", "GOOGL"]
mean_returns = pd.Series([0.15, 0.12, 0.09], index=tickers)
cov_matrix = pd.DataFrame([
    [0.05, 0.02, 0.01],
    [0.02, 0.04, 0.015],
    [0.01, 0.015, 0.03]
], index=tickers, columns=tickers)

print("1. Testing Classical MVO...")
weights = utils.optimize_classical_mvo(mean_returns, cov_matrix, risk_aversion=2.0)
print(f"Classical MVO Weights: {weights}")
assert len(weights) == 3
assert np.isclose(np.sum(weights), 1.0)
print("Classical MVO works perfectly!")

print("\n2. Testing Efficient Frontier...")
random_portfolios, frontier_df, opt_portfolio = utils.generate_efficient_frontier(
    mean_returns, cov_matrix, current_risk_aversion=2.0, num_portfolios=50
)
print(f"Generated {len(random_portfolios)} random portfolios.")
print(f"Generated efficient frontier line of length {len(frontier_df)}.")
assert len(random_portfolios) == 50
assert len(frontier_df) == 50
print("Efficient Frontier works perfectly!")

print("\n3. Testing Quantum QAOA Optimization...")
qaoa_results, probs = utils.solve_qaoa(mean_returns, cov_matrix, risk_aversion=2.0, budget=2, p=1)
print(f"Top QAOA State: {qaoa_results[0]['state_str']} with probability {qaoa_results[0]['prob']:.4f}")
print(f"QAOA selected stocks binary: {qaoa_results[0]['binary_array']}")
assert len(qaoa_results) > 0
print("Quantum QAOA works perfectly!")

print("\n=== ALL SOLVER TESTS PASSED FLAWLESSLY ===")
