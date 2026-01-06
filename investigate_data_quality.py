#!/usr/bin/env python3
"""
Investigate data quality issues - NaN handling and fund inception dates
"""

import pandas as pd
import numpy as np
import os

def investigate_data_quality():
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    funds = pd.read_csv(funds_path, low_memory=False)
    funds['Dates'] = pd.to_datetime(funds['Dates'])
    
    print(f"=== DATA QUALITY INVESTIGATION ===")
    print(f"Total rows: {len(funds)}")
    print(f"Date range: {funds['Dates'].min()} to {funds['Dates'].max()}")
    
    # Check a few funds for NaN patterns
    test_funds = ["TOESCA CI EQUITY", "AAXJ US Equity", "BGFPABA LN EQUITY"]
    
    for fund_ticker in test_funds:
        if fund_ticker in funds.columns:
            print(f"\n=== {fund_ticker} ===")
            
            fund_data = funds[['Dates', fund_ticker]].copy()
            
            # Count NaN values
            total_rows = len(fund_data)
            nan_count = fund_data[fund_ticker].isna().sum()
            valid_count = total_rows - nan_count
            
            print(f"Total observations: {total_rows}")
            print(f"NaN values: {nan_count} ({nan_count/total_rows*100:.1f}%)")
            print(f"Valid values: {valid_count} ({valid_count/total_rows*100:.1f}%)")
            
            # Find first and last valid dates
            valid_data = fund_data.dropna()
            if len(valid_data) > 0:
                first_date = valid_data['Dates'].min()
                last_date = valid_data['Dates'].max()
                print(f"First valid date: {first_date.strftime('%Y-%m-%d')}")
                print(f"Last valid date: {last_date.strftime('%Y-%m-%d')}")
                
                # Check for gaps in the middle
                date_range = pd.date_range(first_date, last_date, freq='D')
                fund_dates = set(valid_data['Dates'].dt.date)
                missing_dates = [d for d in date_range if d.date() not in fund_dates]
                
                print(f"Days in range: {len(date_range)}")
                print(f"Valid trading days: {len(valid_data)}")
                print(f"Missing days in range: {len(missing_dates)}")
                
                # Show first few observations
                print(f"\nFirst 5 valid observations:")
                for i, row in valid_data.head(5).iterrows():
                    print(f"  {row['Dates'].strftime('%Y-%m-%d')}: {row[fund_ticker]:.4f}")
                
                # Show pattern around inception
                print(f"\nData around inception:")
                inception_idx = fund_data[fund_ticker].first_valid_index()
                start_idx = max(0, inception_idx - 3)
                end_idx = min(len(fund_data), inception_idx + 7)
                
                for i in range(start_idx, end_idx):
                    date = fund_data.iloc[i]['Dates']
                    price = fund_data.iloc[i][fund_ticker]
                    status = "VALID" if pd.notna(price) else "NaN"
                    price_str = f"{price:.4f}" if pd.notna(price) else "NaN"
                    marker = " <-- INCEPTION" if i == inception_idx else ""
                    print(f"  {date.strftime('%Y-%m-%d')}: {price_str} ({status}){marker}")

if __name__ == "__main__":
    investigate_data_quality()