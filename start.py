#!/usr/bin/env python3
"""
Simple start script for the Funds Dashboard
"""

import subprocess
import sys
import os

def main():
    """Start the dashboard"""
    print("📈 Starting Funds Dashboard...")
    
    # Check if we're in the right directory
    if not os.path.exists('funds_dashboard.py'):
        print("❌ Error: Please run this script from the ranking-fondos directory")
        return
    
    # Check if data exists
    if not os.path.exists('data/funds_prices.csv'):
        print("❌ Error: Data files not found. Please run 'python convert_data.py' first.")
        return
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'funds_dashboard.py'])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()