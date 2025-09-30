#!/usr/bin/env python3
"""
Test script to verify VaR and CVaR calculations for specific funds
"""

import pandas as pd
import numpy as np
import os

def calculate_performance_metrics_test(funds_df, fund_ticker):
    """Test version of the performance metrics calculation"""
    try:
        if fund_ticker not in funds_df.columns:
            print(f"Fund {fund_ticker} not found in data")
            return None
        
        prices = funds_df[['Dates', fund_ticker]].dropna()
        if len(prices) < 2:
            print(f"Insufficient data for {fund_ticker}")
            return None
        
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        prices['Returns'] = prices[fund_ticker].pct_change()
        
        print(f"\n=== Analysis for {fund_ticker} ===")
        print(f"Data points: {len(prices)}")
        print(f"Date range: {prices['Dates'].min()} to {prices['Dates'].max()}")
        
        # Clean returns
        returns_clean = prices['Returns'].dropna()
        print(f"Valid returns: {len(returns_clean)}")
        
        if len(returns_clean) == 0:
            print("No valid returns found")
            return None
        
        # Basic statistics
        print(f"Mean daily return: {returns_clean.mean():.6f}")
        print(f"Std daily return: {returns_clean.std():.6f}")
        print(f"Min daily return: {returns_clean.min():.6f}")
        print(f"Max daily return: {returns_clean.max():.6f}")
        
        # Volatility (annualized)
        volatility = returns_clean.std() * np.sqrt(252) * 100
        print(f"Annualized volatility: {volatility:.2f}%")
        
        # VaR and CVaR calculations with improved logic
        if len(returns_clean) > 0 and returns_clean.std() > 0:
            # Check for funds with excessive zero returns (low liquidity)
            zero_returns_pct = (returns_clean == 0).sum() / len(returns_clean)
            print(f"Zero returns percentage: {zero_returns_pct:.1%}")
            
            if zero_returns_pct > 0.9:  # More than 90% zero returns
                print("Detected low-liquidity fund - using non-zero returns only")
                # For low-liquidity funds, use only non-zero returns for VaR/CVaR
                non_zero_returns = returns_clean[returns_clean != 0]
                print(f"Non-zero returns: {len(non_zero_returns)}")
                
                if len(non_zero_returns) > 10:  # Need at least 10 non-zero observations
                    daily_var_5 = np.percentile(non_zero_returns, 5)
                    var_5 = daily_var_5 * np.sqrt(252) * 100
                    
                    print(f"Daily VaR 5% (non-zero only): {daily_var_5:.6f}")
                    print(f"Annualized VaR 5%: {var_5:.2f}%")
                    
                    threshold = np.percentile(non_zero_returns, 5)
                    worst_returns = non_zero_returns[non_zero_returns <= threshold]
                    
                    print(f"VaR threshold (non-zero): {threshold:.6f}")
                    print(f"Worst returns count: {len(worst_returns)}")
                    
                    if len(worst_returns) > 0:
                        daily_cvar_5 = worst_returns.mean()
                        cvar_5 = daily_cvar_5 * np.sqrt(252) * 100
                        print(f"Daily CVaR 5%: {daily_cvar_5:.6f}")
                        print(f"Annualized CVaR 5%: {cvar_5:.2f}%")
                    else:
                        cvar_5 = var_5
                        print(f"No worst returns found, using VaR: {cvar_5:.2f}%")
                else:
                    # Not enough non-zero data
                    var_5 = 0
                    cvar_5 = 0
                    daily_var_5 = 0
                    daily_cvar_5 = 0
                    print("Not enough non-zero data for VaR/CVaR calculation")
            else:
                print("Normal liquidity fund - using all returns")
                # Normal calculation for liquid funds
                daily_var_5 = np.percentile(returns_clean, 5)
                var_5 = daily_var_5 * np.sqrt(252) * 100
                
                print(f"Daily VaR 5%: {daily_var_5:.6f}")
                print(f"Annualized VaR 5%: {var_5:.2f}%")
                
                threshold = np.percentile(returns_clean, 5)
                worst_returns = returns_clean[returns_clean <= threshold]
                
                print(f"VaR threshold: {threshold:.6f}")
                print(f"Worst returns count: {len(worst_returns)}")
                
                if len(worst_returns) > 0:
                    daily_cvar_5 = worst_returns.mean()
                    cvar_5 = daily_cvar_5 * np.sqrt(252) * 100
                    print(f"Daily CVaR 5%: {daily_cvar_5:.6f}")
                    print(f"Annualized CVaR 5%: {cvar_5:.2f}%")
                else:
                    cvar_5 = var_5
                    daily_cvar_5 = daily_var_5
                    print(f"No worst returns found, using VaR: {cvar_5:.2f}%")
            
            # Additional validation
            print(f"\nValidation:")
            print(f"5th percentile should be negative for risky assets: {daily_var_5 < 0}")
            if 'daily_cvar_5' in locals():
                print(f"CVaR should be <= VaR: {daily_cvar_5 <= daily_var_5}")
            
            return {
                'Volatility (%)': volatility,
                'VaR 5% (%)': var_5,
                'CVaR 5% (%)': cvar_5,
                'Daily VaR': daily_var_5,
                'Daily CVaR': daily_cvar_5
            }
        else:
            print("Cannot calculate VaR/CVaR: insufficient or zero variance data")
            return None
            
    except Exception as e:
        print(f"Error calculating metrics for {fund_ticker}: {e}")
        return None

def main():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    print("Loading funds data...")
    funds = pd.read_csv(funds_path, low_memory=False)
    funds['Dates'] = pd.to_datetime(funds['Dates'])
    
    print(f"Data loaded: {len(funds)} rows, {len(funds.columns)-1} funds")
    print(f"Date range: {funds['Dates'].min()} to {funds['Dates'].max()}")
    
    # Test the problematic fund
    test_fund = "TOESCA CI EQUITY"
    
    if test_fund in funds.columns:
        print(f"\nTesting {test_fund}...")
        metrics = calculate_performance_metrics_test(funds, test_fund)
        
        if metrics:
            print(f"\n=== FINAL METRICS ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'Daily' in key:
                        print(f"{key}: {value:.6f}")
                    else:
                        print(f"{key}: {value:.2f}%")
    else:
        print(f"Fund {test_fund} not found in data")
        print("Available funds with 'TOESCA' in name:")
        toesca_funds = [col for col in funds.columns if 'TOESCA' in col.upper()]
        for fund in toesca_funds:
            print(f"  - {fund}")

if __name__ == "__main__":
    main()