"""
Reinforcement Learning Trading Environment

A custom trading environment that implements the OpenAI Gym interface
for training RL agents on Bitcoin trading strategies.
"""

# Use gymnasium instead of gym for better NumPy 2.0 compatibility
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    
    The agent can take actions: 0=Hold, 1=Buy, 2=Sell
    The state includes price features, technical indicators, and portfolio state.
    Rewards are based on portfolio value changes and transaction costs.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback_window: int = 20,
        max_position: float = 1.0
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            initial_balance: Starting cash amount
            transaction_cost: Fee percentage per transaction
            lookback_window: Number of past observations to include in state
            max_position: Maximum position size (1.0 = all cash can be used)
        """
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.max_position = max_position
        
        # Prepare feature columns (exclude target if present)
        feature_cols = [col for col in data.columns if col not in ['target', 'model_prediction']]
        self.feature_columns = feature_cols
        
        # Normalize features for better RL training
        self.feature_data = data[feature_cols].fillna(method='ffill').fillna(0)
        self.feature_means = self.feature_data.mean()
        self.feature_stds = self.feature_data.std()
        self.normalized_features = (self.feature_data - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + portfolio state
        n_features = len(self.feature_columns)
        portfolio_features = 3  # cash_ratio, btc_ratio, last_action
        obs_size = n_features * lookback_window + portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Trading state
        self.reset()
        
        logger.info(f"Trading environment initialized")
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Observation space: {self.observation_space.shape}")
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.last_action = 0  # Hold
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        
        # Transaction history
        self.trades = []
        self.portfolio_history = [self.initial_balance]
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Get recent feature data
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        recent_features = self.normalized_features.iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if recent_features.shape[0] < self.lookback_window:
            padding = np.zeros((self.lookback_window - recent_features.shape[0], recent_features.shape[1]))
            recent_features = np.vstack([padding, recent_features])
        
        # Flatten feature history
        feature_obs = recent_features.flatten()
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['Close']
        total_value = self.balance + self.btc_held * current_price
        
        portfolio_obs = np.array([
            self.balance / total_value,  # Cash ratio
            (self.btc_held * current_price) / total_value,  # BTC ratio
            self.last_action / 2.0  # Normalized last action
        ])
        
        # Combine observations
        observation = np.concatenate([feature_obs, portfolio_obs]).astype(np.float32)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Update portfolio tracking
        self.portfolio_value = self.balance + self.btc_held * current_price
        self.portfolio_history.append(self.portfolio_value)
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Prepare info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'price': current_price,
            'action': action,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and calculate reward."""
        reward = 0.0
        
        if action == 1:  # Buy
            if self.balance > 0:
                # Buy as much BTC as possible with available cash
                transaction_cost = self.balance * self.transaction_cost
                available_cash = self.balance - transaction_cost
                btc_to_buy = available_cash / current_price
                
                self.btc_held += btc_to_buy
                self.balance = 0
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'amount': btc_to_buy,
                    'cost': transaction_cost
                })
                
                # Small negative reward for transaction cost
                reward -= transaction_cost / self.initial_balance
                
        elif action == 2:  # Sell
            if self.btc_held > 0:
                # Sell all BTC
                cash_received = self.btc_held * current_price
                transaction_cost = cash_received * self.transaction_cost
                
                self.balance = cash_received - transaction_cost
                btc_sold = self.btc_held
                self.btc_held = 0
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'amount': btc_sold,
                    'cost': transaction_cost
                })
                
                # Small negative reward for transaction cost
                reward -= transaction_cost / self.initial_balance
        
        # Calculate portfolio value change reward
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-1]
            current_value = self.balance + self.btc_held * current_price
            value_change = (current_value - prev_value) / prev_value
            reward += value_change
        
        self.last_action = action
        return reward
    
    def get_portfolio_performance(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if len(self.portfolio_history) < 2:
            return {}
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        max_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        metrics = {
            'total_return': total_return,
            'final_value': self.portfolio_value,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
        }
        
        return metrics
    
    def render(self, mode='human'):
        """Render the current state (for debugging)."""
        current_price = self.data.iloc[self.current_step]['Close'] if self.current_step < len(self.data) else 0
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"BTC Held: {self.btc_held:.6f}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Last Action: {['Hold', 'Buy', 'Sell'][self.last_action]}")
        print("-" * 40)


def test_environment():
    """Test the trading environment with random actions."""
    # Load sample data
    import pandas as pd
    
    # Create sample data if not available
    try:
        data = pd.read_csv('data/btc_featured_data.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        # Create dummy data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 50000,
            'Open': np.random.randn(100).cumsum() + 50000,
            'High': np.random.randn(100).cumsum() + 50000,
            'Low': np.random.randn(100).cumsum() + 50000,
            'Volume': np.random.rand(100) * 1000,
        }, index=dates)
        
        # Add some technical indicators
        data['rsi'] = np.random.rand(100)
        data['macd'] = np.random.randn(100)
        data['bb_upper'] = data['Close'] * 1.1
        data['bb_lower'] = data['Close'] * 0.9
    
    # Initialize environment
    env = TradingEnvironment(data.head(50))  # Use first 50 days for testing
    
    # Run a test episode
    observation = env.reset()
    total_reward = 0
    
    print("Testing Trading Environment...")
    print(f"Initial observation shape: {observation.shape}")
    
    for step in range(20):  # Run for 20 steps
        action = np.random.choice([0, 1, 2])  # Random action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:  # Print every 5 steps
            print(f"Step {step}: Action={['Hold', 'Buy', 'Sell'][action]}, "
                  f"Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if done:
            break
    
    # Print final performance
    performance = env.get_portfolio_performance()
    print("\nFinal Performance:")
    for key, value in performance.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nTotal Reward: {total_reward:.4f}")
    print("Environment test completed successfully!")


if __name__ == "__main__":
    test_environment()