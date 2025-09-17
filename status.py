#!/usr/bin/env python3
"""
NoIQTrader - Project Status Summary
Shows completion status and next steps
"""

import os
import pandas as pd

def main():
    print("🚀 NoIQTrader - AI-Powered Bitcoin Trading System")
    print("=" * 60)
    
    # Check project structure
    project_files = {
        "Data Collection": [
            "src/data_loader.py",
            "data/btc_raw_data.csv"
        ],
        "Feature Engineering": [
            "src/feature_engineer.py", 
            "data/btc_featured_data.csv",
            "data/feature_summary.csv"
        ],
        "Analysis": [
            "notebooks/btc_analysis.ipynb"
        ],
        "Documentation": [
            "README.md",
            "requirements.txt"
        ],
        "Environment": [
            "venv/",
            "start_analysis.sh",
            "process_data.py"
        ]
    }
    
    print("📁 Project Structure Status:")
    for category, files in project_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = "✅" if os.path.exists(file) else "❌"
            print(f"  {exists} {file}")
    
    # Data summary
    if os.path.exists("data/btc_featured_data.csv"):
        print("\n📊 Dataset Summary:")
        data = pd.read_csv("data/btc_featured_data.csv", index_col=0, parse_dates=True)
        print(f"  📈 BTC Data: {len(data):,} trading days")
        print(f"  📅 Period: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  🔧 Features: {len(data.columns)} total")
        print(f"  💾 Size: {os.path.getsize('data/btc_featured_data.csv') / 1024**2:.2f} MB")
        
        # Current price info
        current_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[0]
        total_return = ((current_price / start_price) - 1) * 100
        print(f"  💰 Current BTC: ${current_price:,.2f}")
        print(f"  📈 Total Return: {total_return:.1f}%")
    
    # Technical indicators
    print("\n🔧 Technical Indicators Implemented:")
    indicators = [
        "Moving Averages (10, 50, 200 day)",
        "RSI (Relative Strength Index)", 
        "MACD (Moving Average Convergence Divergence)",
        "Bollinger Bands with position tracking",
        "Multiple volatility measures",
        "Lag features (1, 2, 3 days)",
        "Price momentum indicators",
        "Candlestick pattern features"
    ]
    
    for indicator in indicators:
        print(f"  ✅ {indicator}")
    
    # Phase completion
    print("\n🎯 Phase 1 Status: ✅ COMPLETED")
    print("  ✅ Data Collection & Cleaning")
    print("  ✅ Feature Engineering")
    print("  ✅ Technical Indicators")
    print("  ✅ Data Exploration Notebook")
    print("  ✅ Project Documentation")
    
    # Next steps
    print("\n🚀 Ready for Phase 2:")
    print("  🤖 Machine Learning Models")
    print("     - Logistic Regression")
    print("     - Random Forest")
    print("     - LSTM/Transformer (optional)")
    print("  📊 Paper Trading Simulation")
    print("  🌐 Web Application (Streamlit/Dash)")
    
    print("\n▶️  To start analysis:")
    print("     ./start_analysis.sh")
    print("     # or")
    print("     jupyter notebook notebooks/btc_analysis.ipynb")
    
    print("\n🎉 Phase 1 Complete - Happy Trading! 🎉")

if __name__ == "__main__":
    main()
