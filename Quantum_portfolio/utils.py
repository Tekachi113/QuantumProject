import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Default list of popular S&P 500 stocks
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "V", "DIS", "NFLX"]

def get_stock_data(tickers, start_date, end_date):
    """
    Downloads historical price data (Adjusted Close) from Yahoo Finance.
    Returns a DataFrame containing the historical prices.
    Uses yfinance's native curl_cffi browser impersonation.
    Includes a robust single-ticker fallback loop if the bulk download returns empty.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Step 1: Attempt bulk download (yfinance 1.3.0+ handles curl_cffi internally)
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Check if downloaded data is valid
        prices = pd.DataFrame()
        if isinstance(data, pd.DataFrame) and not data.empty:
            if 'Adj Close' in data:
                prices = data['Adj Close']
            elif 'Close' in data:
                prices = data['Close']
            else:
                prices = data
                
        # Format single ticker case if needed
        if len(tickers) == 1 and not prices.empty:
            ticker = tickers[0]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=ticker)
            elif ticker not in prices.columns:
                prices.columns = [ticker]
                
        # Drop completely NaN columns/rows
        if not prices.empty:
            prices = prices.dropna(how='all')
            prices = prices.ffill().bfill()
        
        # Step 2: Fallback to single-ticker downloads if bulk download failed or is incomplete
        # Multi-ticker endpoints on Yahoo Finance are strictly rate-limited on AWS, but single-ticker endpoints are extremely stable!
        if prices.empty or len(prices.columns) < len(tickers):
            print("Bulk download failed or incomplete. Falling back to single-ticker downloads...")
            prices_list = []
            for ticker in tickers:
                try:
                    ticker_data = yf.download(ticker, start=start_date, end=end_date)
                    if not ticker_data.empty:
                        # Extract close/adj close series
                        if 'Adj Close' in ticker_data:
                            series = ticker_data['Adj Close']
                        elif 'Close' in ticker_data:
                            series = ticker_data['Close']
                        else:
                            series = ticker_data.iloc[:, 0]
                        
                        # Normalize to single Series
                        if isinstance(series, pd.DataFrame):
                            series = series.iloc[:, 0]
                        series.name = ticker
                        prices_list.append(series)
                except Exception as e_single:
                    print(f"Error downloading single ticker {ticker}: {e_single}")
            
            if prices_list:
                prices = pd.concat(prices_list, axis=1)
                prices = prices.dropna(how='all')
                prices = prices.ffill().bfill()
                
        return prices
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_returns(prices_df):
    """
    Calculates daily returns, annualized mean returns, and the covariance matrix.
    """
    # Daily returns
    daily_returns = prices_df.pct_change().dropna()
    
    # Annualized mean returns (assuming 252 trading days)
    mean_returns = daily_returns.mean() * 252
    
    # Annualized covariance matrix (assuming 252 trading days)
    cov_matrix = daily_returns.cov() * 252
    
    return daily_returns, mean_returns, cov_matrix

def optimize_classical_mvo(mean_returns, cov_matrix, risk_aversion):
    """
    Performs Classical Mean-Variance Optimization (MVO) using CVXPY.
    Maximizes: expected_return - 0.5 * risk_aversion * portfolio_variance
    Subject to: sum(w) == 1, w >= 0
    """
    n = len(mean_returns)
    w = cp.Variable(n)
    
    # Values
    mu = mean_returns.values
    Sigma = cov_matrix.values
    
    # Objective
    portfolio_return = mu @ w
    # We use cp.quad_form for the quadratic term w^T Sigma w
    portfolio_variance = cp.quad_form(w, Sigma)
    
    # Objective function
    objective = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
    
    # Constraints: fully invested and long-only (no short selling)
    constraints = [cp.sum(w) == 1, w >= 0]
    
    prob = cp.Problem(objective, constraints)
    
    try:
        # Solve with default solver (cvxpy will automatically pick Clarabel, ECOS, or OSQP)
        prob.solve()
        
        if prob.status == 'optimal':
            weights = w.value
            # Clip numerical noise and normalize
            weights = np.clip(weights, 0, 1)
            weights /= np.sum(weights)
            return weights
        else:
            # Fallback to equal weights if solver fails
            return np.ones(n) / n
    except Exception as e:
        print(f"MVO optimization error: {e}")
        return np.ones(n) / n

def generate_efficient_frontier(mean_returns, cov_matrix, current_risk_aversion, num_portfolios=1000):
    """
    Generates:
    1. A cloud of random portfolios (expected return, risk/volatility, Sharpe Ratio).
    2. The precise Efficient Frontier line by solving MVO for various risk aversion coefficients.
    3. The specific optimal portfolio for the current risk aversion coefficient.
    """
    n = len(mean_returns)
    mu = mean_returns.values
    Sigma = cov_matrix.values
    
    # 1. Generate Random Portfolios for the cloud
    results = []
    # Set seed for reproducible visualization
    np.random.seed(42)
    
    for _ in range(num_portfolios):
        # Dirichlet distribution gives uniform distribution over the simplex sum(w) = 1
        w = np.random.dirichlet(np.ones(n))
        
        # Portfolio return
        p_return = np.dot(w, mu)
        # Portfolio volatility (risk)
        p_vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
        # Sharpe Ratio (assuming risk-free rate = 0.02 or 2%)
        rf = 0.02
        sharpe = (p_return - rf) / p_vol if p_vol > 0 else 0
        
        results.append({
            'return': p_return,
            'volatility': p_vol,
            'sharpe': sharpe,
            'weights': w
        })
        
    random_portfolios = pd.DataFrame(results)
    
    # 2. Generate Efficient Frontier Curve
    frontier_returns = []
    frontier_vols = []
    # Sample risk aversion from very high (low risk) to very low (high return)
    # Using log-spacing for fine resolution at low risk aversion
    risk_aversions = np.logspace(-1.0, 2.5, 50)
    
    for ra in risk_aversions:
        w_opt = optimize_classical_mvo(mean_returns, cov_matrix, ra)
        p_ret = np.dot(w_opt, mu)
        p_vol = np.sqrt(np.dot(w_opt, np.dot(Sigma, w_opt)))
        frontier_returns.append(p_ret)
        frontier_vols.append(p_vol)
        
    frontier_df = pd.DataFrame({
        'return': frontier_returns,
        'volatility': frontier_vols
    }).sort_values('volatility')
    
    # 3. Calculate current optimal portfolio
    w_opt_current = optimize_classical_mvo(mean_returns, cov_matrix, current_risk_aversion)
    opt_ret = np.dot(w_opt_current, mu)
    opt_vol = np.sqrt(np.dot(w_opt_current, np.dot(Sigma, w_opt_current)))
    opt_sharpe = (opt_ret - 0.02) / opt_vol if opt_vol > 0 else 0
    
    opt_portfolio = {
        'return': opt_ret,
        'volatility': opt_vol,
        'sharpe': opt_sharpe,
        'weights': w_opt_current
    }
    
    return random_portfolios, frontier_df, opt_portfolio

# ==========================================
# QUANTUM OPTIMIZATION (QAOA) BACKEND
# ==========================================

def qaoa_circuit(n, p, gamma, beta, h, J):
    """
    Constructs a QAOA circuit for n qubits, p layers.
    Parameters:
    - gamma: list/array of size p (cost parameters)
    - beta: list/array of size p (mixer parameters)
    - h: single-qubit coefficients (size n)
    - J: coupling coefficients (size n x n)
    """
    qc = QuantumCircuit(n)
    
    # Step 1: Initial state - uniform superposition
    for i in range(n):
        qc.h(i)
        
    # Step 2: Apply p layers of QAOA
    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]
        
        # 2a. Cost Hamiltonian: e^{-i g H_C}
        # Single-qubit terms (h_i * Z_i)
        for i in range(n):
            qc.rz(2 * g * h[i], i)
            
        # Two-qubit coupling terms (J_{ij} * Z_i * Z_j)
        for i in range(n):
            for j in range(i + 1, n):
                val = J[i, j]
                if abs(val) > 1e-9:
                    qc.cx(i, j)
                    qc.rz(2 * g * val, j)
                    qc.cx(i, j)
                    
        # 2b. Mixer Hamiltonian: e^{-i b H_B} where H_B = sum X_i
        for i in range(n):
            qc.rx(2 * b, i)
            
    return qc

def compute_expectation(params, n, p, h, J, mean_returns, cov_matrix, risk_aversion, budget, penalty):
    """
    Computes the expectation value of the cost Hamiltonian for a given set of QAOA angles.
    This serves as the objective function for classical optimization.
    """
    gamma = params[:p]
    beta = params[p:]
    
    try:
        # Build circuit and get statevector probabilities
        qc = qaoa_circuit(n, p, gamma, beta, h, J)
        state = Statevector.from_instruction(qc)
        probs = state.probabilities()
        
        expectation = 0.0
        for k in range(2**n):
            if probs[k] < 1e-9:
                continue
            # Convert integer index k to a binary selection vector x
            # x[i] = 1 if qubit i is 1, else 0
            x = np.array([(k >> i) & 1 for i in range(n)])
            
            # Compute classical cost C(x)
            ret = np.dot(mean_returns, x)
            risk = np.dot(x, np.dot(cov_matrix, x))
            budget_penalty = penalty * (np.sum(x) - budget)**2
            
            cost = 0.5 * risk_aversion * risk - ret + budget_penalty
            expectation += probs[k] * cost
            
        return expectation
    except Exception as e:
        print(f"Expectation calculation error: {e}")
        return 99999.0

def solve_qaoa(mean_returns, cov_matrix, risk_aversion, budget, p=1):
    """
    Solves the QUBO portfolio optimization problem using QAOA simulation.
    Finds the optimal angles (gamma, beta) using COBYLA, simulates the final circuit,
    and returns the top portfolio choices and state probabilities.
    """
    n = len(mean_returns)
    mu_val = mean_returns.values if hasattr(mean_returns, 'values') else mean_returns
    cov_val = cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix
    
    # Calculate penalty lambda to enforce budget constraints
    max_cov = np.max(cov_val)
    penalty = max(1.0, 2.0 * risk_aversion * max_cov)
    
    # Formulate Ising Hamiltonian coefficients
    # C(x) = sum_i A_i x_i + sum_{i < j} B_{ij} x_i x_j + const
    # A_i = 0.5 * risk_aversion * Sigma_{ii} - mu_i + lambda * (1 - 2*B)
    # B_{ij} = risk_aversion * Sigma_{ij} + 2 * lambda
    #
    # Ising: x_i = (1 - Z_i)/2
    # J_{ij} = B_{ij}/4 = (risk_aversion * Sigma_{ij} + 2 * lambda) / 4
    # h_i = -A_i/2 - sum_{j != i} J_{ij}
    
    A = np.zeros(n)
    for i in range(n):
        A[i] = 0.5 * risk_aversion * cov_val[i, i] - mu_val[i] + penalty * (1.0 - 2.0 * budget)
        
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = (risk_aversion * cov_val[i, j] + 2.0 * penalty) / 4.0
            J[i, j] = val
            J[j, i] = val
            
    h = np.zeros(n)
    for i in range(n):
        sum_J = sum(J[i, j] for j in range(n) if j != i)
        h[i] = -A[i] / 2.0 - sum_J
        
    # Initial parameters
    init_params = np.ones(2 * p) * 0.1
    
    # Optimization
    res = minimize(
        compute_expectation,
        init_params,
        args=(n, p, h, J, mu_val, cov_val, risk_aversion, budget, penalty),
        method='COBYLA',
        options={'maxiter': 60}  # Low maxiter for snappy UI response
    )
    
    opt_params = res.x
    opt_gamma = opt_params[:p]
    opt_beta = opt_params[p:]
    
    # Compute final statevector
    qc = qaoa_circuit(n, p, opt_gamma, opt_beta, h, J)
    state = Statevector.from_instruction(qc)
    probs = state.probabilities()
    
    # Sort indices by probability
    top_indices = np.argsort(probs)[::-1]
    
    results = []
    # We want to display the top 8 states (since 2^N can be large, we just show the most probable)
    for idx in top_indices[:12]:
        x = np.array([(idx >> i) & 1 for i in range(n)])
        # Bitstring display in standard order (MSB to LSB), i.e., stock 0 is leftmost or rightmost?
        # Let's keep it big-endian: left is stock index 0, right is stock index n-1.
        # This matches the way we usually read arrays! So x[::-1] would make the first stock index 0 at the rightmost?
        # Actually, if x[0] is the state of stock 0, then:
        # stock_0 stock_1 ... stock_n-1:
        # we can show it as "".join(str(val) for val in x) so that index 0 is leftmost!
        state_str = "".join(str(val) for val in x)
        
        # Calculate metrics of this state
        num_selected = int(np.sum(x))
        if num_selected > 0:
            ret = np.dot(mu_val, x)
            # Volatility of equal weight allocation among selected stocks
            # weights = x / sum(x)
            w_eq = x / float(num_selected)
            risk = np.sqrt(np.dot(w_eq, np.dot(cov_val, w_eq)))
            ret_eq = np.dot(mu_val, w_eq)
        else:
            ret = 0.0
            risk = 0.0
            ret_eq = 0.0
            
        results.append({
            'state_str': state_str,
            'binary_array': x,
            'prob': probs[idx],
            'return': ret_eq,  # Return of equal weighted selection
            'risk': risk,      # Volatility of equal weighted selection
            'num_selected': num_selected
        })
        
    return results, probs
