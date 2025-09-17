"""
Feature Engineering Module for NoIQTrader
Implements technical indicators and feature creation for BTC trading analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating technical indicators and trading features
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        logger.info("Initializing Feature Engineer")
    
    def add_moving_averages(self, data: pd.DataFrame, windows: list = [10, 50, 200]) -> pd.DataFrame:
        """
        Add moving averages to the dataset
        
        Args:
            data (pd.DataFrame): Input data with Close prices
            windows (list): List of window sizes for moving averages
            
        Returns:
            pd.DataFrame: Data with moving averages added
        """
        logger.info(f"Adding moving averages with windows: {windows}")
        
        data_copy = data.copy()
        
        for window in windows:
            col_name = f"MA{window}"
            data_copy[col_name] = data_copy['Close'].rolling(window=window).mean()
            
            # Add ratio features (price relative to MA)
            data_copy[f"Close_to_{col_name}_ratio"] = data_copy['Close'] / data_copy[col_name]
        
        return data_copy
    
    def add_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI)
        
        Args:
            data (pd.DataFrame): Input data with Close prices
            window (int): RSI calculation window
            
        Returns:
            pd.DataFrame: Data with RSI added
        """
        logger.info(f"Adding RSI with window: {window}")
        
        data_copy = data.copy()
        
        # Calculate price changes
        delta = data_copy['Close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        data_copy['RSI'] = rsi
        
        # Add RSI signal features
        data_copy['RSI_oversold'] = (data_copy['RSI'] < 30).astype(int)
        data_copy['RSI_overbought'] = (data_copy['RSI'] > 70).astype(int)
        
        return data_copy
    
    def add_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD)
        
        Args:
            data (pd.DataFrame): Input data with Close prices
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            pd.DataFrame: Data with MACD indicators added
        """
        logger.info(f"Adding MACD with parameters: fast={fast}, slow={slow}, signal={signal}")
        
        data_copy = data.copy()
        
        # Calculate EMAs
        ema_fast = data_copy['Close'].ewm(span=fast).mean()
        ema_slow = data_copy['Close'].ewm(span=slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Add to dataframe
        data_copy['MACD'] = macd_line
        data_copy['MACD_signal'] = signal_line
        data_copy['MACD_histogram'] = histogram
        
        # Add MACD signal features
        data_copy['MACD_bullish'] = (data_copy['MACD'] > data_copy['MACD_signal']).astype(int)
        data_copy['MACD_cross_up'] = ((data_copy['MACD'] > data_copy['MACD_signal']) & 
                                     (data_copy['MACD'].shift(1) <= data_copy['MACD_signal'].shift(1))).astype(int)
        
        return data_copy
    
    def add_bollinger_bands(self, data: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Add Bollinger Bands
        
        Args:
            data (pd.DataFrame): Input data with Close prices
            window (int): Moving average window
            std_dev (int): Standard deviation multiplier
            
        Returns:
            pd.DataFrame: Data with Bollinger Bands added
        """
        logger.info(f"Adding Bollinger Bands with window: {window}, std_dev: {std_dev}")
        
        data_copy = data.copy()
        
        # Calculate moving average and standard deviation
        ma = data_copy['Close'].rolling(window=window).mean()
        std = data_copy['Close'].rolling(window=window).std()
        
        # Calculate bands
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        # Add to dataframe
        data_copy['BB_upper'] = upper_band
        data_copy['BB_middle'] = ma
        data_copy['BB_lower'] = lower_band
        
        # Calculate Bollinger Band features
        data_copy['BB_width'] = (upper_band - lower_band) / ma
        data_copy['BB_position'] = (data_copy['Close'] - lower_band) / (upper_band - lower_band)
        
        # Add signal features
        data_copy['BB_squeeze'] = (data_copy['BB_width'] < data_copy['BB_width'].rolling(20).mean()).astype(int)
        data_copy['BB_breakout_upper'] = (data_copy['Close'] > data_copy['BB_upper']).astype(int)
        data_copy['BB_breakout_lower'] = (data_copy['Close'] < data_copy['BB_lower']).astype(int)
        
        return data_copy
    
    def add_volatility_features(self, data: pd.DataFrame, windows: list = [10, 20, 30]) -> pd.DataFrame:
        """
        Add volatility-based features
        
        Args:
            data (pd.DataFrame): Input data with OHLC prices
            windows (list): List of windows for volatility calculation
            
        Returns:
            pd.DataFrame: Data with volatility features added
        """
        logger.info(f"Adding volatility features with windows: {windows}")
        
        data_copy = data.copy()
        
        # Calculate daily returns
        data_copy['daily_return'] = data_copy['Close'].pct_change()
        
        # Calculate True Range for volatility
        data_copy['prev_close'] = data_copy['Close'].shift(1)
        data_copy['true_range'] = np.maximum(
            data_copy['High'] - data_copy['Low'],
            np.maximum(
                abs(data_copy['High'] - data_copy['prev_close']),
                abs(data_copy['Low'] - data_copy['prev_close'])
            )
        )
        
        # Calculate rolling volatilities
        for window in windows:
            # Standard deviation of returns
            data_copy[f'volatility_{window}d'] = data_copy['daily_return'].rolling(window=window).std()
            
            # Average True Range
            data_copy[f'ATR_{window}d'] = data_copy['true_range'].rolling(window=window).mean()
            
            # High-Low percentage
            data_copy[f'HL_pct_{window}d'] = ((data_copy['High'] - data_copy['Low']) / data_copy['Close']).rolling(window=window).mean()
        
        # Calculate Garman-Klass volatility (more accurate for daily data)
        data_copy['GK_volatility'] = np.sqrt(
            0.5 * np.log(data_copy['High'] / data_copy['Low'])**2 - 
            (2 * np.log(2) - 1) * np.log(data_copy['Close'] / data_copy['Open'])**2
        )
        
        # Rolling Garman-Klass volatility
        data_copy['GK_volatility_20d'] = data_copy['GK_volatility'].rolling(window=20).mean()
        
        # Clean up temporary columns
        data_copy.drop(['prev_close'], axis=1, inplace=True)
        
        return data_copy
    
    def add_lag_features(self, data: pd.DataFrame, lags: list = [1, 2, 3], columns: list = ['Close']) -> pd.DataFrame:
        """
        Add lag features for specified columns
        
        Args:
            data (pd.DataFrame): Input data
            lags (list): List of lag periods
            columns (list): List of columns to create lags for
            
        Returns:
            pd.DataFrame: Data with lag features added
        """
        logger.info(f"Adding lag features for columns: {columns}, lags: {lags}")
        
        data_copy = data.copy()
        
        for col in columns:
            if col in data_copy.columns:
                for lag in lags:
                    data_copy[f'{col}_lag_{lag}'] = data_copy[col].shift(lag)
        
        return data_copy
    
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features
        
        Args:
            data (pd.DataFrame): Input data with OHLC prices
            
        Returns:
            pd.DataFrame: Data with price features added
        """
        logger.info("Adding price-based features")
        
        data_copy = data.copy()
        
        # Price ratios
        data_copy['open_close_ratio'] = data_copy['Open'] / data_copy['Close']
        data_copy['high_close_ratio'] = data_copy['High'] / data_copy['Close']
        data_copy['low_close_ratio'] = data_copy['Low'] / data_copy['Close']
        
        # Price changes
        data_copy['price_change'] = data_copy['Close'] - data_copy['Open']
        data_copy['price_change_pct'] = (data_copy['Close'] - data_copy['Open']) / data_copy['Open']
        
        # Candlestick patterns (simplified)
        data_copy['is_green'] = (data_copy['Close'] > data_copy['Open']).astype(int)
        data_copy['is_red'] = (data_copy['Close'] < data_copy['Open']).astype(int)
        data_copy['is_doji'] = (abs(data_copy['Close'] - data_copy['Open']) / data_copy['Close'] < 0.001).astype(int)
        
        # Body and shadow sizes
        data_copy['body_size'] = abs(data_copy['Close'] - data_copy['Open'])
        data_copy['upper_shadow'] = data_copy['High'] - np.maximum(data_copy['Open'], data_copy['Close'])
        data_copy['lower_shadow'] = np.minimum(data_copy['Open'], data_copy['Close']) - data_copy['Low']
        
        # Volume features
        data_copy['volume_ma_10'] = data_copy['Volume'].rolling(window=10).mean()
        data_copy['volume_ratio'] = data_copy['Volume'] / data_copy['volume_ma_10']
        
        return data_copy
    
    def add_momentum_features(self, data: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Add momentum-based features
        
        Args:
            data (pd.DataFrame): Input data with Close prices
            windows (list): List of windows for momentum calculation
            
        Returns:
            pd.DataFrame: Data with momentum features added
        """
        logger.info(f"Adding momentum features with windows: {windows}")
        
        data_copy = data.copy()
        
        for window in windows:
            # Price momentum (rate of change)
            data_copy[f'momentum_{window}d'] = (data_copy['Close'] / data_copy['Close'].shift(window) - 1) * 100
            
            # Rolling correlation with time (trend strength)
            data_copy[f'trend_strength_{window}d'] = data_copy['Close'].rolling(window=window).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) == window else np.nan
            )
        
        return data_copy
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicators and features
        
        Args:
            data (pd.DataFrame): Raw BTC OHLCV data
            
        Returns:
            pd.DataFrame: Data with all features added
        """
        logger.info("Creating all technical features...")
        
        # Start with the original data
        featured_data = data.copy()
        
        # Add all features
        featured_data = self.add_moving_averages(featured_data)
        featured_data = self.add_rsi(featured_data)
        featured_data = self.add_macd(featured_data)
        featured_data = self.add_bollinger_bands(featured_data)
        featured_data = self.add_volatility_features(featured_data)
        featured_data = self.add_lag_features(featured_data)
        featured_data = self.add_price_features(featured_data)
        featured_data = self.add_momentum_features(featured_data)
        
        # Log feature creation summary
        original_cols = len(data.columns)
        new_cols = len(featured_data.columns)
        logger.info(f"Feature engineering completed: {original_cols} -> {new_cols} columns")
        logger.info(f"Added {new_cols - original_cols} new features")
        
        return featured_data
    
    def get_feature_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary of all features
        
        Args:
            data (pd.DataFrame): Data with features
            
        Returns:
            pd.DataFrame: Feature summary
        """
        summary = pd.DataFrame({
            'Feature': data.columns,
            'Missing_Values': data.isnull().sum().values,
            'Missing_Percentage': (data.isnull().sum() / len(data) * 100).values,
            'Data_Type': data.dtypes.values
        })
        
        return summary.sort_values('Missing_Percentage', ascending=False)


def main():
    """
    Main function for testing the feature engineer
    """
    # Import data loader
    import sys
    sys.path.append('.')
    from data_loader import BTCDataLoader
    
    # Load BTC data
    loader = BTCDataLoader()
    btc_data = loader.get_btc_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create all features
    featured_data = engineer.create_all_features(btc_data)
    
    # Display summary
    print(f"Original shape: {btc_data.shape}")
    print(f"Featured shape: {featured_data.shape}")
    print(f"Added {featured_data.shape[1] - btc_data.shape[1]} features")
    
    # Show feature summary
    feature_summary = engineer.get_feature_summary(featured_data)
    print("\nFeature Summary:")
    print(feature_summary.head(20))
    
    # Show sample of featured data
    print("\nSample of featured data:")
    print(featured_data.tail())


if __name__ == "__main__":
    main()
