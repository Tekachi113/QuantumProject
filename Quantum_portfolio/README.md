# ⚛️ Quantum Portfolio Optimization

This project demonstrates the application of **Quantum Computing** to the classic Markowitz Mean-Variance Optimization problem in quantitative finance. 

By formulating the portfolio allocation as a **QUBO** (Quadratic Unconstrained Binary Optimization) problem, we can solve it using the **QAOA** (Quantum Approximate Optimization Algorithm) on simulators and real IBM Quantum hardware.

---

## 🚀 Features

- **End-to-end Pipeline:** Fetches real S&P500 historical data via `yfinance`, calculates expected returns and covariance.
- **Classical Benchmark:** Computes the Global Minimum Variance (GMV), Max Sharpe portfolios, and Efficient Frontier using `CVXPY`.
- **Quantum Formulation:** Automatically maps the continuous Markowitz problem to a binary QUBO format, then to an Ising Hamiltonian.
- **QAOA Circuit Construction:** Parameterized QAOA circuits with customizable depth ($p=1, 2, 3$) using `Qiskit`.
- **IBM Quantum Hardware Results:** Analyzes raw outputs from IBM's 127-qubit `ibm_brisbane` processor.
- **Error Mitigation (ZNE):** Zero Noise Extrapolation using gate folding to mitigate hardware noise.
- **Interactive UI:** A beautiful, responsive multi-page web app built with `Streamlit`.

---

## 🛠️ Technology Stack

- **Quantum:** `Qiskit`, `Qiskit Aer`, IBM Quantum Platform
- **Optimization:** `CVXPY`, `scipy.optimize`
- **Data & Math:** `pandas`, `numpy`, `yfinance`
- **Frontend / Vis:** `Streamlit`, `Plotly`, `Matplotlib`

---

## ⚙️ Installation

1. Clone this repository.
2. Ensure you have Python 3.10+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run the Web App

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The app is divided into 4 pages:
1. **Khám Phá Dữ Liệu:** Data explorer for S&P500 stocks.
2. **Tối Ưu Cổ Điển:** Markowitz MVO and Efficient Frontier.
3. **Tối Ưu Lượng Tử:** QAOA solver simulation.
4. **So Sánh Tổng Hợp:** Side-by-side comparison including IBM hardware runs and ZNE noise mitigation.

---

## 📂 Project Structure

```
Quantum portfolio/
├── app.py                      # Streamlit entry point (Landing Page)
├── app/
│   ├── components/
│   │   └── sidebar.py          # Shared UI components
│   └── pages/
│       ├── 1_📊_Du_Lieu.py     # Data Page
│       ├── 2_📈_Co_Dien_MVO.py # Classical MVO Page
│       ├── 3_⚛️_Luong_Tu_QAOA.py # Quantum QAOA Page
│       └── 4_🏆_So_Sanh.py     # Comparison & Hardware Page
├── src/
│   ├── classical/              # MVO Solvers and metrics
│   ├── data/                   # Data fetcher and processor
│   └── quantum/                # QUBO, circuit builder, and QAOA optimizer
├── notebooks/
│   └── quantum_portfolio_optimization.ipynb # Full academic notebook
├── reports/
│   └── quantum_portfolio_report.md          # IEEE-style scientific report
├── tests/                      # Unit tests
├── utils.py                    # Helper functions for the Streamlit app
├── config.yaml                 # Core configuration
└── requirements.txt            # Python dependencies
```

---

## 📜 Academic Deliverables

- **Jupyter Notebook:** Located in `notebooks/quantum_portfolio_optimization.ipynb`. Contains the full narrative from data fetching to quantum simulation and hardware error mitigation.
- **Scientific Report:** Located in `reports/quantum_portfolio_report.md`. Formatted in IEEE style, discussing the potential and limitations of NISQ-era quantum computers in finance.

---
*Built for the Quantum Computing in Finance coursework.*
