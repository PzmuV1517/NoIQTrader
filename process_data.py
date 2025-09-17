#!/usr/bin/env python3
"""
NoIQTrader - Complete Data Processing Script
Generates the full featured dataset for modeling
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_loader import BTCDataLoader
from feature_engineer import FeatureEngineer

def main():
    """Main function to process and save complete dataset"""
    print(" NoIQTrader - Phase 1 Data Processing")
    print("=" * 50)
    
    # Initialize components
    print(" Initializing data loader and feature engineer...")
    data_loader = BTCDataLoader(data_dir='data')
    feature_engineer = FeatureEngineer()
    
    # Load BTC data
    print(" Loading BTC-USD historical data...")
    btc_data = data_loader.get_btc_data()
    print(f" Loaded {len(btc_data)} days of data from {btc_data.index[0].date()} to {btc_data.index[-1].date()}")
    
    # Create all features
    print(" Engineering technical features...")
    featured_data = feature_engineer.create_all_features(btc_data)
    print(f" Created {len(featured_data.columns)} features ({len(featured_data.columns) - len(btc_data.columns)} new)")
    
    # Clean data (remove rows with too many missing values)
    print(" Cleaning data...")
    threshold = 0.8 * len(featured_data.columns)
    clean_data = featured_data.dropna(thresh=int(threshold))
    print(f" Retained {len(clean_data)}/{len(featured_data)} rows ({len(clean_data)/len(featured_data)*100:.1f}%)")
    
    # Save processed data
    output_path = 'data/btc_featured_data.csv'
    clean_data.to_csv(output_path)
    print(f" Saved featured dataset: {output_path}")
    print(f"   Shape: {clean_data.shape}")
    print(f"   Size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    
    # Create feature summary
    feature_summary = feature_engineer.get_feature_summary(clean_data)
    feature_summary.to_csv('data/feature_summary.csv', index=False)
    print(f" Saved feature summary: data/feature_summary.csv")
    
    # Display final summary
    print("\n Phase 1 Completion Summary:")
    print(f"   Data Period: {clean_data.index[0].date()} to {clean_data.index[-1].date()}")
    print(f"   Trading Days: {len(clean_data):,}")
    print(f"   Features: {len(clean_data.columns)}")
    print(f"   Technical Indicators: MA, RSI, MACD, Bollinger Bands, Volatility")
    print(f"   Lag Features: 1, 2, 3 day lags")
    print(f"   Price Features: Returns, ratios, momentum")
    print(f"   Data Quality: {(1 - clean_data.isnull().sum().sum() / (len(clean_data) * len(clean_data.columns)))*100:.1f}% complete")
    
    # Show current BTC price info
    current_price = clean_data['Close'].iloc[-1]
    price_change_5y = ((current_price / clean_data['Close'].iloc[0]) - 1) * 100
    print(f"\n Current BTC Info:")
    print(f"   Price: ${current_price:,.2f}")
    print(f"   5-Year Return: {price_change_5y:.1f}%")
    
    # Show recent volatility
    if 'volatility_20d' in clean_data.columns:
        recent_vol = clean_data['volatility_20d'].iloc[-1] * 100
        print(f"   20-Day Volatility: {recent_vol:.2f}%")
    
    print(f"\n Ready for Phase 2: Machine Learning Modeling!")
    print(f"   Use 'jupyter notebook notebooks/btc_analysis.ipynb' to explore the data")

if __name__ == "__main__":
    main()
