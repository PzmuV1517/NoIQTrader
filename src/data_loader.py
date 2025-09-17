"""
Data Loader Module for NoIQTrader
Handles Bitcoin data collection from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BTCDataLoader:
    """
    Bitcoin data loader class for fetching and caching BTC-USD data
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Directory to store cached data
        """
        self.data_dir = data_dir
        self.symbol = "BTC-USD"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_btc_data(self, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch Bitcoin historical data from Yahoo Finance
        
        Args:
            period (str): Time period to fetch (5y for 5 years)
            interval (str): Data interval (1d for daily)
            
        Returns:
            pd.DataFrame: Bitcoin OHLCV data
        """
        try:
            logger.info(f"Fetching BTC-USD data for period: {period}")
            
            # Create ticker object
            btc_ticker = yf.Ticker(self.symbol)
            
            # Fetch historical data
            data = btc_ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError("No data retrieved from Yahoo Finance")
                
            # Clean column names (remove any extra spaces)
            data.columns = data.columns.str.strip()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove timezone info and ensure datetime index
            data.index = pd.to_datetime(data.index).tz_localize(None)
            
            # Sort by date
            data = data.sort_index()
            
            logger.info(f"Successfully fetched {len(data)} days of BTC data")
            logger.info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching BTC data: {str(e)}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the fetched data
        
        Args:
            data (pd.DataFrame): Raw BTC data
            
        Returns:
            pd.DataFrame: Cleaned BTC data
        """
        logger.info("Cleaning BTC data...")
        
        # Create a copy to avoid modifying original
        cleaned_data = data.copy()
        
        # Check for missing values
        missing_values = cleaned_data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values}")
            
            # Forward fill missing values (carry forward last known value)
            cleaned_data = cleaned_data.fillna(method='ffill')
            
            # If there are still missing values at the beginning, backward fill
            cleaned_data = cleaned_data.fillna(method='bfill')
        
        # Check for any remaining missing values
        remaining_missing = cleaned_data.isnull().sum()
        if remaining_missing.any():
            logger.warning(f"Still have missing values after cleaning: {remaining_missing}")
        
        # Validate data integrity
        self._validate_data(cleaned_data)
        
        logger.info("Data cleaning completed successfully")
        return cleaned_data
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate data integrity
        
        Args:
            data (pd.DataFrame): Data to validate
        """
        # Check for negative values (shouldn't happen with price data)
        if (data[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            raise ValueError("Negative price values found in data")
        
        # Check that High >= Low for each day
        if (data['High'] < data['Low']).any():
            raise ValueError("Invalid data: High price less than Low price found")
        
        # Check that High >= Open and High >= Close
        if (data['High'] < data['Open']).any() or (data['High'] < data['Close']).any():
            logger.warning("Some High prices are less than Open/Close prices")
        
        # Check that Low <= Open and Low <= Close
        if (data['Low'] > data['Open']).any() or (data['Low'] > data['Close']).any():
            logger.warning("Some Low prices are greater than Open/Close prices")
        
        # Check for duplicate dates
        if data.index.duplicated().any():
            raise ValueError("Duplicate dates found in data")
        
        logger.info("Data validation passed")
    
    def save_data(self, data: pd.DataFrame, filename: str = "btc_raw_data.csv") -> str:
        """
        Save data to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Filename for the saved data
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str = "btc_raw_data.csv") -> Optional[pd.DataFrame]:
        """
        Load data from CSV file
        
        Args:
            filename (str): Filename to load
            
        Returns:
            pd.DataFrame or None: Loaded data or None if file doesn't exist
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"Loading data from {filepath}")
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return data
        else:
            logger.info(f"File {filepath} does not exist")
            return None
    
    def get_btc_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get BTC data - either from cache or fetch fresh data
        
        Args:
            force_refresh (bool): If True, fetch fresh data regardless of cache
            
        Returns:
            pd.DataFrame: BTC OHLCV data
        """
        # Try to load from cache first
        if not force_refresh:
            cached_data = self.load_data()
            if cached_data is not None:
                # Check if cached data is recent (within last day)
                last_date = pd.to_datetime(cached_data.index[-1])
                today = datetime.now()
                
                if (today - last_date).days <= 1:
                    logger.info("Using cached data (recent)")
                    return cached_data
                else:
                    logger.info("Cached data is outdated, fetching fresh data")
        
        # Fetch fresh data
        raw_data = self.fetch_btc_data()
        cleaned_data = self.clean_data(raw_data)
        
        # Save to cache
        self.save_data(cleaned_data)
        
        return cleaned_data


def main():
    """
    Main function for testing the data loader
    """
    # Initialize data loader
    loader = BTCDataLoader()
    
    # Fetch and clean data
    btc_data = loader.get_btc_data()
    
    # Display basic info
    print(f"BTC Data Shape: {btc_data.shape}")
    print(f"Date Range: {btc_data.index[0].date()} to {btc_data.index[-1].date()}")
    print(f"Columns: {list(btc_data.columns)}")
    print("\nFirst 5 rows:")
    print(btc_data.head())
    print("\nLast 5 rows:")
    print(btc_data.tail())
    print("\nBasic Statistics:")
    print(btc_data.describe())


if __name__ == "__main__":
    main()
