#!/usr/bin/env python3
"""
Configuration example for the Funds Dashboard
Copy this file to config.py and modify as needed
"""

# Default scoring weights (should sum to 100)
DEFAULT_WEIGHTS = {
    'YTD Return (%)': 20,
    'MTD Return (%)': 0,
    'Monthly Return (%)': 0,
    '1Y Return (%)': 25,
    '2024 Return (%)': 15,
    '2023 Return (%)': 10,
    '2022 Return (%)': 0,
    'Max Drawdown (%)': 15,
    'Volatility (%)': 15,
    'VaR 5% (%)': 0,
    'CVaR 5% (%)': 0
}

# Risk-free rate for Sharpe ratio calculation (annual)
RISK_FREE_RATE = 0.02  # 2%

# Number of trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252

# Default date ranges for analysis
DEFAULT_ANALYSIS_PERIOD_DAYS = 365  # 1 year

# Chart configuration
CHART_COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', 
    '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6366f1'
]

# Performance metrics configuration
METRICS_CONFIG = {
    'var_confidence_level': 0.05,  # 5% VaR
    'min_observations_for_var': 20,  # Minimum data points for VaR calculation
    'volatility_annualization_factor': 252**0.5,
    'return_annualization_factor': 252
}

# Efficient frontier configuration
FRONTIER_CONFIG = {
    'num_portfolios': 50,  # Number of points on efficient frontier
    'solver': 'ECOS',  # CVXPY solver
    'max_weight': 1.0,  # Maximum weight per asset (100%)
    'min_weight': 0.0,  # Minimum weight per asset (0% - long only)
}

# Display configuration
DISPLAY_CONFIG = {
    'max_funds_in_chart': 10,
    'max_funds_in_frontier': 10,
    'top_funds_to_show': 20,
    'decimal_places': 2,
    'percentage_format': '{:.2f}%'
}

# Data validation
DATA_VALIDATION = {
    'min_price_observations': 2,
    'min_return_observations': 10,
    'max_missing_data_ratio': 0.5  # 50% missing data threshold
}

# Export configuration
EXPORT_CONFIG = {
    'pdf_page_size': 'A4',
    'csv_encoding': 'utf-8',
    'date_format': '%Y-%m-%d',
    'timestamp_format': '%Y%m%d_%H%M'
}