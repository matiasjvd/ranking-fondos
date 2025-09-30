#!/usr/bin/env python3
"""
Analyze non-zero returns for TOESCA fund
"""

import pandas as pd
import numpy as np
import os

def analyze_nonzero_returns():
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
    
    # Get non-zero returns
    nonzero_returns = returns_clean[returns_clean != 0]
    
    print(f"=== NON-ZERO RETURNS ANALYSIS ===")
    print(f"Total returns: {len(returns_clean)}")
    print(f"Non-zero returns: {len(nonzero_returns)} ({len(nonzero_returns)/len(returns_clean)*100:.1f}%)")
    
    if len(nonzero_returns) > 0:
        print(f"\nNon-zero returns statistics:")
        print(f"  Min: {nonzero_returns.min():.6f}")
        print(f"  Max: {nonzero_returns.max():.6f}")
        print(f"  Mean: {nonzero_returns.mean():.6f}")
        print(f"  Std: {nonzero_returns.std():.6f}")
        
        print(f"\nPercentiles of non-zero returns:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(nonzero_returns, p)
            print(f"  {p}th percentile: {val:.6f}")
        
        print(f"\nAll non-zero returns:")
        for i, (idx, ret) in enumerate(nonzero_returns.items()):
            date = prices.iloc[idx]['Dates'].strftime('%Y-%m-%d')
            price_before = prices.iloc[idx-1][fund_ticker] if idx > 0 else np.nan
            price_after = prices.iloc[idx][fund_ticker]
            print(f"  {i+1:2d}. {date}: {ret:10.6f} ({price_before:.2f} -> {price_after:.2f})")
    
    # Calculate proper VaR/CVaR for this type of fund
    print(f"\n=== PROPER VAR/CVAR CALCULATION ===")
    
    # For funds with mostly zero returns, we should use all returns
    var_5 = np.percentile(returns_clean, 5)
    cvar_5 = returns_clean[returns_clean <= var_5].mean()
    
    print(f"VaR 5% (all returns): {var_5:.6f} ({var_5*100:.2f}%)")
    print(f"CVaR 5% (all returns): {cvar_5:.6f} ({cvar_5*100:.2f}%)")
    
    # Alternative: use only when there are actual price movements
    if len(nonzero_returns) >= 10:  # Need minimum sample size
        var_5_nonzero = np.percentile(nonzero_returns, 5)
        cvar_5_nonzero = nonzero_returns[nonzero_returns <= var_5_nonzero].mean()
        
        print(f"\nAlternative calculation (non-zero returns only):")
        print(f"VaR 5% (non-zero only): {var_5_nonzero:.6f} ({var_5_nonzero*100:.2f}%)")
        print(f"CVaR 5% (non-zero only): {cvar_5_nonzero:.6f} ({cvar_5_nonzero*100:.2f}%)")
        print(f"Note: This represents risk only on days when prices actually move")

if __name__ == "__main__":
    analyze_nonzero_returns()