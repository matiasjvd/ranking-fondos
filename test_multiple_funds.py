#!/usr/bin/env python3
"""
Test the corrected calculation with multiple funds
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to Python path to import the dashboard functions
sys.path.append('/Users/matias/Desktop/Proyectos/ranking-fondos')

def test_multiple_funds():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    funds_df = pd.read_csv(funds_path, low_memory=False)
    funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
    
    # Test different types of funds
    test_funds = [
        "TOESCA CI EQUITY",      # Low liquidity, late inception
        "AAXJ US Equity",        # Liquid, medium inception
        "BGFPABA LN EQUITY",     # Liquid, early inception
    ]
    
    print(f"=== TESTING MULTIPLE FUNDS ===")
    
    for fund_ticker in test_funds:
        if fund_ticker not in funds_df.columns:
            print(f"\n{fund_ticker}: NOT FOUND")
            continue
            
        print(f"\n=== {fund_ticker} ===")
        
        # Simulate the corrected calculation
        prices = funds_df[['Dates', fund_ticker]].copy()
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        
        # Find inception
        first_valid_idx = prices[fund_ticker].first_valid_index()
        if first_valid_idx is None:
            print("No valid data")
            continue
            
        inception_date = prices.iloc[first_valid_idx]['Dates']
        print(f"Inception: {inception_date.strftime('%Y-%m-%d')}")
        
        # Process data from inception
        prices_clean = prices.iloc[first_valid_idx:].reset_index(drop=True)
        prices_clean[fund_ticker] = prices_clean[fund_ticker].ffill()
        prices_clean = prices_clean.dropna()
        
        if len(prices_clean) < 2:
            print("Insufficient data")
            continue
            
        prices_clean['Returns'] = prices_clean[fund_ticker].pct_change()
        returns_clean = prices_clean['Returns'].dropna()
        
        print(f"Data points: {len(prices_clean)}")
        print(f"Returns: {len(returns_clean)}")
        
        # Calculate metrics
        if len(returns_clean) > 0:
            # Zero returns analysis
            zero_returns = (returns_clean == 0).sum()
            zero_pct = zero_returns / len(returns_clean) * 100
            print(f"Zero returns: {zero_pct:.1f}%")
            
            # VaR/CVaR
            daily_var_5 = np.percentile(returns_clean, 5)
            var_5 = daily_var_5 * np.sqrt(252) * 100
            
            threshold = np.percentile(returns_clean, 5)
            worst_returns = returns_clean[returns_clean <= threshold]
            if len(worst_returns) > 0:
                daily_cvar_5 = worst_returns.mean()
                cvar_5 = daily_cvar_5 * np.sqrt(252) * 100
            else:
                cvar_5 = var_5
            
            # Volatility
            volatility = returns_clean.std() * np.sqrt(252) * 100
            
            # YTD Return
            current_date = prices_clean['Dates'].max()
            current_year = current_date.year
            ytd_start = pd.to_datetime(f'{current_year}-01-01')
            ytd_data = prices_clean[prices_clean['Dates'] >= ytd_start]
            ytd_return = ((ytd_data[fund_ticker].iloc[-1] / ytd_data[fund_ticker].iloc[0]) - 1) * 100 if len(ytd_data) > 1 else 0
            
            print(f"VaR 5%: {var_5:.2f}%")
            print(f"CVaR 5%: {cvar_5:.2f}%")
            print(f"Volatility: {volatility:.2f}%")
            print(f"YTD Return: {ytd_return:.2f}%")

if __name__ == "__main__":
    test_multiple_funds()