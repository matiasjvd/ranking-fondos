#!/usr/bin/env python3
"""
Quick start script for the Funds Dashboard
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'cvxpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_files():
    """Check if data files exist"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    required_files = ['funds_prices.csv', 'funds_dictionary.csv']
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    return missing_files

def run_conversion():
    """Run data conversion script"""
    print("🔄 Converting Excel data to CSV...")
    try:
        subprocess.check_call([sys.executable, 'convert_data.py'])
        print("✅ Data conversion completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting data: {e}")
        return False

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Funds Dashboard...")
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'funds_dashboard.py'])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main function to set up and run the dashboard"""
    print("📈 Funds Dashboard - Quick Start")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('funds_dashboard.py'):
        print("❌ Error: funds_dashboard.py not found!")
        print("Please run this script from the ranking-fondos directory")
        return
    
    # Check requirements
    missing_packages = check_requirements()
    if missing_packages:
        print(f"📦 Missing packages: {', '.join(missing_packages)}")
        if input("Install missing packages? (y/n): ").lower() == 'y':
            if not install_requirements():
                return
        else:
            print("❌ Cannot run dashboard without required packages")
            return
    else:
        print("✅ All required packages are installed")
    
    # Check data files
    missing_files = check_data_files()
    if missing_files:
        print(f"📊 Missing data files: {', '.join(missing_files)}")
        if os.path.exists('convert_data.py'):
            if input("Run data conversion? (y/n): ").lower() == 'y':
                if not run_conversion():
                    print("❌ Data conversion failed. Please check the original Excel files.")
                    return
            else:
                print("❌ Cannot run dashboard without data files")
                return
        else:
            print("❌ convert_data.py not found. Please ensure data files exist in the data/ directory")
            return
    else:
        print("✅ All data files are available")
    
    # Launch dashboard
    print("\n🎯 Everything is ready!")
    print("The dashboard will open in your default web browser")
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 40)
    
    run_dashboard()

if __name__ == "__main__":
    main()