
import pandas as pd
from metrics_calculator import calculate_individual_fund_metrics
import os

def test_metrics():
    # Load data
    data_dir = 'data'
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    funds = pd.read_csv(funds_path)
    
    # Check max date
    funds['Dates'] = pd.to_datetime(funds['Dates'])
    print(f"Max date in data: {funds['Dates'].max()}")
    
    # Test a few funds
    test_tickers = funds.columns[2:5] # Skip index and Dates
    for ticker in test_tickers:
        print(f"\nTesting metrics for: {ticker}")
        metrics = calculate_individual_fund_metrics(funds, ticker)
        if metrics:
            for key, value in metrics.items():
                if 'Return (%)' in key:
                    print(f"  {key}: {value:.2f}%")
        else:
            print(f"  No metrics calculated for {ticker}")

if __name__ == "__main__":
    test_metrics()
