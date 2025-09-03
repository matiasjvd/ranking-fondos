#!/usr/bin/env python3
"""
Test script to verify dashboard functionality
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    try:
        # Test CSV loading
        data_dir = 'data'
        funds = pd.read_csv(os.path.join(data_dir, 'funds_prices.csv'))
        etf_dict = pd.read_csv(os.path.join(data_dir, 'funds_dictionary.csv'))
        
        print(f"✅ Funds data: {len(funds)} rows, {len(funds.columns)} columns")
        print(f"✅ Dictionary: {len(etf_dict)} entries")
        
        # Test date parsing
        funds['Dates'] = pd.to_datetime(funds['Dates'])
        print(f"✅ Date parsing successful")
        
        return funds, etf_dict
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None

def test_performance_calculation(funds_df):
    """Test performance metrics calculation"""
    print("\n🧪 Testing performance calculation...")
    
    try:
        # Get a sample fund
        available_funds = [col for col in funds_df.columns if col != 'Dates']
        sample_fund = available_funds[0]
        
        # Test basic calculations
        prices = funds_df[['Dates', sample_fund]].dropna()
        if len(prices) < 2:
            print(f"❌ Insufficient data for {sample_fund}")
            return False
        
        prices['Returns'] = prices[sample_fund].pct_change()
        
        # Test YTD calculation
        current_date = prices['Dates'].max()
        current_year = current_date.year
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = prices[prices['Dates'] >= ytd_start]
        
        if len(ytd_data) > 1:
            ytd_return = ((ytd_data[sample_fund].iloc[-1] / ytd_data[sample_fund].iloc[0]) - 1) * 100
            print(f"✅ YTD calculation successful: {ytd_return:.2f}%")
        
        # Test volatility calculation
        volatility = prices['Returns'].std() * np.sqrt(252) * 100
        print(f"✅ Volatility calculation successful: {volatility:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance calculation failed: {e}")
        return False

def test_fund_matching(funds_df, etf_dict):
    """Test fund matching between price data and dictionary"""
    print("\n🧪 Testing fund matching...")
    
    try:
        available_funds = set([col for col in funds_df.columns if col != 'Dates'])
        dict_tickers = set(etf_dict['Ticker'].astype(str))
        
        matches = available_funds.intersection(dict_tickers)
        print(f"✅ Direct matches: {len(matches)} funds")
        
        # Test fuzzy matching
        fuzzy_matches = 0
        for fund in available_funds:
            for dict_ticker in dict_tickers:
                if fund.replace(' ', '').upper() == dict_ticker.replace(' ', '').upper():
                    fuzzy_matches += 1
                    break
        
        print(f"✅ Total matchable funds: {fuzzy_matches}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fund matching failed: {e}")
        return False

def test_filtering(etf_dict):
    """Test filtering functionality"""
    print("\n🧪 Testing filtering...")
    
    try:
        # Test unique categories
        regions = etf_dict['Geografia'].unique()
        asset_classes = etf_dict['Asset Class'].unique()
        sectors = etf_dict['Sector'].unique()
        
        print(f"✅ Regions: {len(regions)} ({', '.join(regions[:3])}...)")
        print(f"✅ Asset Classes: {len(asset_classes)} ({', '.join(asset_classes[:3])}...)")
        print(f"✅ Sectors: {len(sectors)} ({', '.join(sectors[:3])}...)")
        
        return True
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Dashboard Tests")
    print("=" * 40)
    
    # Test data loading
    funds_df, etf_dict = test_data_loading()
    if funds_df is None or etf_dict is None:
        print("\n❌ Critical error: Data loading failed")
        return False
    
    # Test performance calculation
    if not test_performance_calculation(funds_df):
        print("\n❌ Performance calculation tests failed")
        return False
    
    # Test fund matching
    if not test_fund_matching(funds_df, etf_dict):
        print("\n❌ Fund matching tests failed")
        return False
    
    # Test filtering
    if not test_filtering(etf_dict):
        print("\n❌ Filtering tests failed")
        return False
    
    print("\n" + "=" * 40)
    print("✅ All tests passed! Dashboard is ready to run.")
    print("\nTo start the dashboard, run:")
    print("  python run_dashboard.py")
    print("  or")
    print("  streamlit run funds_dashboard.py")
    
    return True

if __name__ == "__main__":
    main()