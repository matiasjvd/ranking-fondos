#!/usr/bin/env python3
"""
Debug script to investigate TOESCA data issues
"""

import pandas as pd
import numpy as np
import os

def debug_toesca_data():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    funds = pd.read_csv(funds_path, low_memory=False)
    funds['Dates'] = pd.to_datetime(funds['Dates'])
    
    fund_ticker = "TOESCA CI EQUITY"
    
    if fund_ticker not in funds.columns:
        print(f"Fund {fund_ticker} not found")
        return
    
    # Get fund data
    prices = funds[['Dates', fund_ticker]].dropna()
    prices = prices.sort_values('Dates').reset_index(drop=True)
    prices['Returns'] = prices[fund_ticker].pct_change()
    
    returns_clean = prices['Returns'].dropna()
    
    print(f"=== TOESCA DATA INVESTIGATION ===")
    print(f"Total returns: {len(returns_clean)}")
    
    # Check for zero returns
    zero_returns = (returns_clean == 0).sum()
    print(f"Zero returns: {zero_returns} ({zero_returns/len(returns_clean)*100:.1f}%)")
    
    # Check for near-zero returns
    near_zero = (np.abs(returns_clean) < 1e-10).sum()
    print(f"Near-zero returns (< 1e-10): {near_zero}")
    
    # Percentile analysis
    percentiles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"\nPercentile analysis:")
    for p in percentiles:
        val = np.percentile(returns_clean, p)
        print(f"  {p}th percentile: {val:.8f}")
    
    # Show distribution of returns around 5th percentile
    p5 = np.percentile(returns_clean, 5)
    print(f"\nReturns around 5th percentile ({p5:.8f}):")
    
    # Count returns at exactly the 5th percentile
    exact_p5 = (returns_clean == p5).sum()
    print(f"Returns exactly at 5th percentile: {exact_p5}")
    
    # Show some actual return values
    print(f"\nFirst 20 returns:")
    for i, ret in enumerate(returns_clean.head(20)):
        print(f"  {i+1}: {ret:.8f}")
    
    print(f"\nLast 20 returns:")
    for i, ret in enumerate(returns_clean.tail(20)):
        print(f"  {len(returns_clean)-19+i}: {ret:.8f}")
    
    # Show sorted returns around 5th percentile
    sorted_returns = returns_clean.sort_values()
    p5_index = int(len(sorted_returns) * 0.05)
    
    print(f"\nSorted returns around 5th percentile (index {p5_index}):")
    start_idx = max(0, p5_index - 10)
    end_idx = min(len(sorted_returns), p5_index + 10)
    
    for i in range(start_idx, end_idx):
        marker = " <-- 5th percentile" if i == p5_index else ""
        print(f"  {i}: {sorted_returns.iloc[i]:.8f}{marker}")
    
    # Check price data for issues
    print(f"\n=== PRICE DATA ANALYSIS ===")
    print(f"Price range: {prices[fund_ticker].min():.4f} to {prices[fund_ticker].max():.4f}")
    
    # Check for repeated prices
    price_changes = prices[fund_ticker].diff().dropna()
    zero_price_changes = (price_changes == 0).sum()
    print(f"Days with no price change: {zero_price_changes} ({zero_price_changes/len(price_changes)*100:.1f}%)")
    
    # Show some price data
    print(f"\nFirst 10 price observations:")
    for i in range(min(10, len(prices))):
        date = prices.iloc[i]['Dates'].strftime('%Y-%m-%d')
        price = prices.iloc[i][fund_ticker]
        ret = prices.iloc[i]['Returns'] if i > 0 else np.nan
        print(f"  {date}: {price:.4f} (return: {ret:.8f})")

if __name__ == "__main__":
    debug_toesca_data()