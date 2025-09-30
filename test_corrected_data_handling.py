#!/usr/bin/env python3
"""
Test the corrected data handling with forward fill and proper inception dates
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

def test_corrected_data_handling():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    funds_df = pd.read_csv(funds_path, low_memory=False)
    funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
    
    # Test with TOESCA (low liquidity fund with late inception)
    fund_ticker = "TOESCA CI EQUITY"
    
    print(f"=== TESTING CORRECTED DATA HANDLING ===")
    print(f"Fund: {fund_ticker}")
    
    # Simulate the corrected calculation
    if fund_ticker not in funds_df.columns:
        print("Fund not found")
        return
    
    # Get all data for this fund
    prices = funds_df[['Dates', fund_ticker]].copy()
    prices['Dates'] = pd.to_datetime(prices['Dates'])
    prices = prices.sort_values('Dates').reset_index(drop=True)
    
    print(f"Total rows in dataset: {len(prices)}")
    print(f"Date range: {prices['Dates'].min()} to {prices['Dates'].max()}")
    
    # Find the first valid price (fund inception date)
    first_valid_idx = prices[fund_ticker].first_valid_index()
    print(f"First valid index: {first_valid_idx}")
    
    if first_valid_idx is not None:
        inception_date = prices.iloc[first_valid_idx]['Dates']
        print(f"Fund inception date: {inception_date.strftime('%Y-%m-%d')}")
        
        # Only use data from inception date onwards
        prices_from_inception = prices.iloc[first_valid_idx:].reset_index(drop=True)
        print(f"Rows from inception: {len(prices_from_inception)}")
        
        # Check NaN before forward fill
        nan_count_before = prices_from_inception[fund_ticker].isna().sum()
        print(f"NaN values after inception (before ffill): {nan_count_before}")
        
        # Forward fill prices to handle missing data after inception
        prices_from_inception[fund_ticker] = prices_from_inception[fund_ticker].fillna(method='ffill')
        
        # Check NaN after forward fill
        nan_count_after = prices_from_inception[fund_ticker].isna().sum()
        print(f"NaN values after forward fill: {nan_count_after}")
        
        # Remove any remaining NaN
        prices_clean = prices_from_inception.dropna()
        print(f"Final clean rows: {len(prices_clean)}")
        
        if len(prices_clean) >= 2:
            prices_clean['Returns'] = prices_clean[fund_ticker].pct_change()
            returns_clean = prices_clean['Returns'].dropna()
            
            print(f"\n=== PERFORMANCE METRICS ===")
            print(f"Returns calculated: {len(returns_clean)}")
            
            # Check zero returns
            zero_returns = (returns_clean == 0).sum()
            zero_pct = zero_returns / len(returns_clean) * 100
            print(f"Zero returns: {zero_returns} ({zero_pct:.1f}%)")
            
            # Calculate VaR/CVaR with corrected data
            if len(returns_clean) > 0:
                daily_var_5 = np.percentile(returns_clean, 5)
                var_5 = daily_var_5 * np.sqrt(252) * 100
                
                threshold = np.percentile(returns_clean, 5)
                worst_returns = returns_clean[returns_clean <= threshold]
                if len(worst_returns) > 0:
                    daily_cvar_5 = worst_returns.mean()
                    cvar_5 = daily_cvar_5 * np.sqrt(252) * 100
                else:
                    cvar_5 = var_5
                
                print(f"VaR 5%: {var_5:.2f}%")
                print(f"CVaR 5%: {cvar_5:.2f}%")
                
                # Volatility
                volatility = returns_clean.std() * np.sqrt(252) * 100
                print(f"Volatility: {volatility:.2f}%")
                
                # Show data quality around inception
                print(f"\n=== DATA AROUND INCEPTION ===")
                print("First 10 observations from inception:")
                for i in range(min(10, len(prices_clean))):
                    date = prices_clean.iloc[i]['Dates']
                    price = prices_clean.iloc[i][fund_ticker]
                    ret = prices_clean.iloc[i]['Returns'] if i > 0 else np.nan
                    ret_str = f"{ret:.6f}" if pd.notna(ret) else "NaN"
                    print(f"  {date.strftime('%Y-%m-%d')}: {price:.4f} (return: {ret_str})")

if __name__ == "__main__":
    test_corrected_data_handling()