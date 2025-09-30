#!/usr/bin/env python3
"""
Test the corrected VaR/CVaR calculation
"""

import pandas as pd
import numpy as np
import os

def test_corrected_calculation():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    funds = pd.read_csv(funds_path, low_memory=False)
    funds['Dates'] = pd.to_datetime(funds['Dates'])
    
    fund_ticker = "TOESCA CI EQUITY"
    
    # Get fund data
    prices = funds[['Dates', fund_ticker]].dropna()
    prices = prices.sort_values('Dates').reset_index(drop=True)
    prices['Returns'] = prices[fund_ticker].pct_change()
    
    returns_clean = prices['Returns'].dropna()
    
    print(f"=== CORRECTED VAR/CVAR CALCULATION ===")
    print(f"Fund: {fund_ticker}")
    print(f"Total returns: {len(returns_clean)}")
    
    # Check zero returns percentage
    zero_returns_pct = (returns_clean == 0).sum() / len(returns_clean)
    print(f"Zero returns: {zero_returns_pct*100:.1f}%")
    
    if len(returns_clean) > 0:
        # Standard VaR/CVaR calculation using all returns
        daily_var_5 = np.percentile(returns_clean, 5)
        var_5 = daily_var_5 * np.sqrt(252) * 100
        
        threshold = np.percentile(returns_clean, 5)
        worst_returns = returns_clean[returns_clean <= threshold]
        if len(worst_returns) > 0:
            daily_cvar_5 = worst_returns.mean()
            cvar_5 = daily_cvar_5 * np.sqrt(252) * 100
        else:
            cvar_5 = var_5
        
        print(f"\nCorrected Results:")
        print(f"Daily VaR 5%: {daily_var_5:.6f}")
        print(f"Daily CVaR 5%: {daily_cvar_5:.6f}")
        print(f"Annualized VaR 5%: {var_5:.2f}%")
        print(f"Annualized CVaR 5%: {cvar_5:.2f}%")
        
        # Additional statistics
        print(f"\nAdditional info:")
        print(f"5th percentile value: {np.percentile(returns_clean, 5):.6f}")
        print(f"Number of returns at/below 5th percentile: {len(worst_returns)}")
        print(f"Mean of worst returns: {worst_returns.mean():.6f}")
        
        # Compare with a liquid fund for reference
        print(f"\n=== COMPARISON WITH LIQUID FUND ===")
        liquid_fund = "AAXJ US Equity"  # Pick a liquid fund
        
        if liquid_fund in funds.columns:
            liquid_prices = funds[['Dates', liquid_fund]].dropna()
            liquid_prices = liquid_prices.sort_values('Dates').reset_index(drop=True)
            liquid_prices['Returns'] = liquid_prices[liquid_fund].pct_change()
            liquid_returns = liquid_prices['Returns'].dropna()
            
            liquid_zero_pct = (liquid_returns == 0).sum() / len(liquid_returns)
            liquid_var_5 = np.percentile(liquid_returns, 5) * np.sqrt(252) * 100
            liquid_worst = liquid_returns[liquid_returns <= np.percentile(liquid_returns, 5)]
            liquid_cvar_5 = liquid_worst.mean() * np.sqrt(252) * 100
            
            print(f"Liquid fund: {liquid_fund}")
            print(f"Zero returns: {liquid_zero_pct*100:.1f}%")
            print(f"VaR 5%: {liquid_var_5:.2f}%")
            print(f"CVaR 5%: {liquid_cvar_5:.2f}%")

if __name__ == "__main__":
    test_corrected_calculation()