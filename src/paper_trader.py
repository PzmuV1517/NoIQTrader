"""
Paper Trading Simulator for NoIQTrader

This module implements a virtual trading portfolio that simulates trades
based on machine learning model predictions without using real money.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os


class PaperTradingPortfolio:
    """
    Simulates a virtual trading portfolio for backtesting ML trading strategies.
    
    Features:
    - Virtual cash and BTC holdings
    - Trade execution based on model signals
    - Performance tracking (PnL, returns, drawdown)
    - Transaction cost simulation
    - Detailed trade history
    """
    
    def __init__(self, initial_cash: float = 10000, 
                 transaction_fee: float = 0.001,
                 min_trade_amount: float = 100):
        """
        Initialize paper trading portfolio.
        
        Args:
            initial_cash: Starting cash amount ($10,000 default)
            transaction_fee: Trading fee as percentage (0.1% default)
            min_trade_amount: Minimum trade size in USD
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.btc_holdings = 0.0
        self.transaction_fee = transaction_fee
        self.min_trade_amount = min_trade_amount
        
        # Portfolio tracking
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_cash
        
    def get_portfolio_value(self, btc_price: float) -> float:
        """Calculate total portfolio value at current BTC price."""
        return self.cash + (self.btc_holdings * btc_price)
    
    def execute_trade(self, signal: int, btc_price: float, date: str, 
                     confidence: float = 0.0) -> Dict:
        """
        Execute a trade based on the trading signal.
        
        Args:
            signal: Trading signal (-1: Sell, 0: Hold, 1: Buy)
            btc_price: Current BTC price
            date: Trade date
            confidence: Model confidence (0-1)
            
        Returns:
            Dictionary with trade details
        """
        portfolio_value_before = self.get_portfolio_value(btc_price)
        trade_info = {
            'date': date,
            'signal': signal,
            'btc_price': btc_price,
            'confidence': confidence,
            'action': 'Hold',
            'amount': 0,
            'fees': 0,
            'cash_before': self.cash,
            'btc_before': self.btc_holdings,
            'portfolio_value_before': portfolio_value_before
        }
        
        if signal == 1:  # Buy signal
            trade_info.update(self._execute_buy(btc_price))
        elif signal == -1:  # Sell signal
            trade_info.update(self._execute_sell(btc_price))
        # signal == 0 means Hold (no action)
        
        # Update portfolio value after trade
        portfolio_value_after = self.get_portfolio_value(btc_price)
        trade_info.update({
            'cash_after': self.cash,
            'btc_after': self.btc_holdings,
            'portfolio_value_after': portfolio_value_after
        })
        
        # Record trade
        self.trade_history.append(trade_info)
        self.total_trades += 1
        
        # Update performance tracking
        self._update_performance_metrics(portfolio_value_after)
        
        return trade_info
    
    def _execute_buy(self, btc_price: float) -> Dict:
        """Execute a buy order using available cash."""
        if self.cash < self.min_trade_amount:
            return {'action': 'Hold', 'amount': 0, 'fees': 0, 
                   'reason': 'Insufficient cash for minimum trade'}
        
        # Use all available cash for buying (aggressive strategy)
        trade_amount = self.cash
        fees = trade_amount * self.transaction_fee
        net_amount = trade_amount - fees
        btc_bought = net_amount / btc_price
        
        # Update holdings
        self.cash = 0
        self.btc_holdings += btc_bought
        
        return {
            'action': 'Buy',
            'amount': trade_amount,
            'btc_amount': btc_bought,
            'fees': fees,
            'reason': 'Buy signal executed'
        }
    
    def _execute_sell(self, btc_price: float) -> Dict:
        """Execute a sell order using available BTC."""
        if self.btc_holdings == 0:
            return {'action': 'Hold', 'amount': 0, 'fees': 0,
                   'reason': 'No BTC holdings to sell'}
        
        # Sell all BTC holdings (aggressive strategy)
        btc_to_sell = self.btc_holdings
        gross_amount = btc_to_sell * btc_price
        
        if gross_amount < self.min_trade_amount:
            return {'action': 'Hold', 'amount': 0, 'fees': 0,
                   'reason': 'BTC holdings below minimum trade amount'}
        
        fees = gross_amount * self.transaction_fee
        net_amount = gross_amount - fees
        
        # Update holdings
        self.cash += net_amount
        self.btc_holdings = 0
        
        return {
            'action': 'Sell',
            'amount': gross_amount,
            'btc_amount': btc_to_sell,
            'fees': fees,
            'reason': 'Sell signal executed'
        }
    
    def _update_performance_metrics(self, current_value: float):
        """Update performance tracking metrics."""
        # Track peak value and drawdown
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value
        
        current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record portfolio value
        self.portfolio_history.append({
            'value': current_value,
            'cash': self.cash,
            'btc': self.btc_holdings,
            'peak_value': self.peak_portfolio_value,
            'drawdown': current_drawdown
        })
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary."""
        if not self.portfolio_history:
            return {'error': 'No trading history available'}
        
        current_value = self.portfolio_history[-1]['value']
        total_return = (current_value - self.initial_cash) / self.initial_cash
        
        # Calculate daily returns if we have enough data
        daily_returns = []
        if len(self.portfolio_history) > 1:
            for i in range(1, len(self.portfolio_history)):
                prev_val = self.portfolio_history[i-1]['value']
                curr_val = self.portfolio_history[i]['value']
                if prev_val > 0:
                    daily_return = (curr_val - prev_val) / prev_val
                    daily_returns.append(daily_return)
        
        # Calculate additional metrics
        winning_trades = sum(1 for trade in self.trade_history 
                           if trade.get('portfolio_value_after', 0) > trade.get('portfolio_value_before', 0))
        
        total_fees = sum(trade.get('fees', 0) for trade in self.trade_history)
        
        return {
            'initial_value': self.initial_cash,
            'final_value': current_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'absolute_profit': current_value - self.initial_cash,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': self.total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / max(self.total_trades, 1) * 100,
            'total_fees': total_fees,
            'daily_returns': daily_returns,
            'volatility': np.std(daily_returns) * np.sqrt(252) if daily_returns else 0,  # Annualized
            'sharpe_ratio': (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)) if daily_returns and np.std(daily_returns) > 0 else 0,
            'current_cash': self.cash,
            'current_btc': self.btc_holdings
        }


class BacktestEngine:
    """
    Backtesting engine that runs paper trading simulation on historical data.
    """
    
    def __init__(self, data_path: str, models_path: str = None):
        """
        Initialize backtesting engine.
        
        Args:
            data_path: Path to CSV file with historical data and predictions
            models_path: Path to saved models (optional)
        """
        self.data_path = data_path
        self.models_path = models_path
        self.data = None
        self.portfolio = None
        self.backtest_results = None
        
    def load_data(self) -> pd.DataFrame:
        """Load historical data with features and predictions."""
        print("📊 Loading historical data for backtesting...")
        
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        print(f"✅ Loaded {len(self.data)} days of data")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
        return self.data
    
    def run_backtest(self, prediction_column: str = 'target', 
                    initial_cash: float = 10000,
                    start_date: str = None,
                    end_date: str = None) -> Dict:
        """
        Run backtesting simulation using model predictions.
        
        Args:
            prediction_column: Column name containing trading signals
            initial_cash: Starting portfolio value
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dictionary containing backtest results
        """
        if self.data is None:
            self.load_data()
        
        # Filter data by date range if specified
        test_data = self.data.copy()
        if start_date:
            test_data = test_data[test_data.index >= start_date]
        if end_date:
            test_data = test_data[test_data.index <= end_date]
        
        if len(test_data) == 0:
            raise ValueError("No data available for specified date range")
        
        print(f"🚀 Running backtest from {test_data.index[0].date()} to {test_data.index[-1].date()}")
        print(f"Total days: {len(test_data)}")
        
        # Initialize portfolio
        self.portfolio = PaperTradingPortfolio(initial_cash=initial_cash)
        
        # Execute trades day by day
        for date, row in test_data.iterrows():
            signal = row.get(prediction_column, 0)
            btc_price = row['Close']
            confidence = row.get('confidence', 0.5)  # Default confidence if not available
            
            # Execute trade
            trade_info = self.portfolio.execute_trade(
                signal=signal,
                btc_price=btc_price,
                date=date.strftime('%Y-%m-%d'),
                confidence=confidence
            )
        
        # Generate results
        performance = self.portfolio.get_performance_summary()
        
        # Add benchmark comparison (Buy and Hold)
        buy_hold_return = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0]
        performance['benchmark_return'] = buy_hold_return * 100
        performance['vs_benchmark'] = performance['total_return_pct'] - (buy_hold_return * 100)
        
        self.backtest_results = {
            'performance': performance,
            'portfolio_history': self.portfolio.portfolio_history,
            'trade_history': self.portfolio.trade_history,
            'data': test_data
        }
        
        print(f"\n✅ Backtest completed!")
        print(f"Final Portfolio Value: ${performance['final_value']:,.2f}")
        print(f"Total Return: {performance['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.1f}%")
        print(f"vs Buy & Hold: {performance['vs_benchmark']:+.2f}%")
        
        return self.backtest_results
    
    def get_trade_markers(self) -> Tuple[List, List, List]:
        """
        Extract trade markers for plotting.
        
        Returns:
            Tuple of (buy_dates, buy_prices, sell_dates, sell_prices, hold_info)
        """
        if not self.backtest_results:
            return [], [], []
        
        buy_dates, buy_prices = [], []
        sell_dates, sell_prices = [], []
        
        for trade in self.portfolio.trade_history:
            if trade['action'] == 'Buy':
                buy_dates.append(pd.to_datetime(trade['date']))
                buy_prices.append(trade['btc_price'])
            elif trade['action'] == 'Sell':
                sell_dates.append(pd.to_datetime(trade['date']))
                sell_prices.append(trade['btc_price'])
        
        return buy_dates, buy_prices, sell_dates, sell_prices
    
    def save_results(self, output_path: str):
        """Save backtest results to file."""
        if not self.backtest_results:
            raise ValueError("No backtest results to save. Run backtest first.")
        
        # Convert datetime objects to strings for JSON serialization
        results_to_save = {
            'performance': self.backtest_results['performance'],
            'trade_history': [
                {k: (v.strftime('%Y-%m-%d') if isinstance(v, (pd.Timestamp, datetime)) else v)
                 for k, v in trade.items()}
                for trade in self.backtest_results['trade_history']
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f"💾 Backtest results saved to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Test the paper trading system
    print("🧪 Testing Paper Trading System...")
    
    # Load data
    data_path = "data/btc_featured_data.csv"
    if os.path.exists(data_path):
        engine = BacktestEngine(data_path)
        engine.load_data()
        
        # Run a simple backtest using target variable as predictions
        results = engine.run_backtest(
            prediction_column='target',
            initial_cash=10000,
            start_date='2024-01-01'  # Use recent data for testing
        )
        
        print("\n📈 Sample Trade History (last 5 trades):")
        for trade in engine.portfolio.trade_history[-5:]:
            print(f"{trade['date']}: {trade['action']} at ${trade['btc_price']:,.2f} "
                  f"(Portfolio: ${trade['portfolio_value_after']:,.2f})")
        
    else:
        print(f"❌ Data file not found at {data_path}")
        print("Please run the feature engineering pipeline first.")
