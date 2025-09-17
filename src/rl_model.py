"""
Reinforcement Learning Model Integration

Integrates RL trading models with the existing NoIQTrader system,
including training, prediction, and model persistence.
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Import RL components
try:
    from .rl_trading_env import TradingEnvironment
    from .rl_dqn_agent import DQNAgent
except ImportError:
    from rl_trading_env import TradingEnvironment
    from rl_dqn_agent import DQNAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLTradingModel:
    """
    Reinforcement Learning Trading Model that integrates with the existing system.
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize RL Trading Model.
        
        Args:
            lookback_window: Number of past observations for state
            initial_balance: Starting balance for trading
            transaction_cost: Transaction fee percentage
            learning_rate: Learning rate for DQN
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
        """
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Model components
        self.env = None
        self.agent = None
        self.is_trained = False
        self.training_history = []
        
        # Model metadata
        self.model_info = {
            'model_type': 'DQN_RL',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'lookback_window': lookback_window,
                'initial_balance': initial_balance,
                'transaction_cost': transaction_cost,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'epsilon_decay': epsilon_decay
            }
        }
        
        logger.info("RL Trading Model initialized")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for RL training.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            Prepared DataFrame
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                logger.warning(f"Missing required column: {col}")
        
        # Fill missing values
        data_clean = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        data_clean = data_clean.dropna()
        
        logger.info(f"Prepared {len(data_clean)} samples for RL training")
        return data_clean
    
    def initialize_environment(self, data: pd.DataFrame):
        """Initialize the trading environment with data."""
        self.env = TradingEnvironment(
            data=data,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
            lookback_window=self.lookback_window
        )
        
        # Initialize agent with correct state size
        state_size = self.env.observation_space.shape[0]
        self.agent = DQNAgent(
            state_size=state_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay
        )
        
        logger.info(f"Environment initialized with state size: {state_size}")
    
    def train(
        self, 
        data: pd.DataFrame, 
        episodes: int = 100,
        save_freq: int = 20,
        early_stopping: bool = True,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            data: Training data
            episodes: Number of training episodes
            save_freq: Frequency to save model during training
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            
        Returns:
            Training results
        """
        if self.env is None or self.agent is None:
            data_clean = self.prepare_data(data)
            self.initialize_environment(data_clean)
        
        logger.info(f"Starting RL training for {episodes} episodes")
        
        episode_rewards = []
        episode_portfolio_values = []
        best_performance = float('-inf')
        episodes_without_improvement = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step = 0
            
            while True:
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.train_step()
                
                state = next_state
                total_reward += reward
                step += 1
                
                if done:
                    break
            
            # Get portfolio performance
            performance = self.env.get_portfolio_performance()
            final_value = performance.get('final_value', self.initial_balance)
            
            episode_rewards.append(total_reward)
            episode_portfolio_values.append(final_value)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_value = np.mean(episode_portfolio_values[-10:])
                
                logger.info(f"Episode {episode}/{episodes}: "
                          f"Avg Reward: {avg_reward:.4f}, "
                          f"Avg Portfolio: ${avg_value:.2f}, "
                          f"Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model periodically
            if episode > 0 and episode % save_freq == 0:
                self._save_checkpoint(episode)
            
            # Early stopping check
            if early_stopping:
                current_performance = np.mean(episode_portfolio_values[-5:])
                if current_performance > best_performance:
                    best_performance = current_performance
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                if episodes_without_improvement >= patience:
                    logger.info(f"Early stopping at episode {episode}")
                    break
        
        # Training completed
        self.is_trained = True
        
        # Calculate final metrics
        final_metrics = {
            'episodes_trained': episode + 1,
            'avg_reward': np.mean(episode_rewards),
            'best_portfolio_value': max(episode_portfolio_values),
            'final_portfolio_value': episode_portfolio_values[-1],
            'total_return': (episode_portfolio_values[-1] - self.initial_balance) / self.initial_balance,
            'agent_stats': self.agent.get_training_stats()
        }
        
        self.training_history.append(final_metrics)
        
        logger.info("RL training completed")
        logger.info(f"Final metrics: {final_metrics}")
        
        return final_metrics
    
    def predict(self, data: pd.DataFrame, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make trading predictions using the trained RL agent.
        
        Args:
            data: Current market data
            return_probabilities: Whether to return action probabilities
            
        Returns:
            Prediction results
        """
        if not self.is_trained or self.agent is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Initialize environment for prediction
        if len(data) < self.lookback_window:
            logger.warning(f"Insufficient data for prediction. Need at least {self.lookback_window} samples")
            return {'action': 0, 'confidence': 0.0, 'action_name': 'Hold'}
        
        pred_env = TradingEnvironment(
            data=data.tail(self.lookback_window + 10),  # Use recent data
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
            lookback_window=self.lookback_window
        )
        
        # Get current state
        state = pred_env.reset()
        
        # Get action from agent (no exploration)
        action = self.agent.act(state, training=False)
        
        # Get Q-values for confidence estimation
        q_values = self.agent.predict(state)
        
        # Calculate confidence as softmax of Q-values
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        probabilities = exp_q / np.sum(exp_q)
        confidence = probabilities[action]
        
        action_names = ['Hold', 'Buy', 'Sell']
        
        result = {
            'action': action,
            'action_name': action_names[action],
            'confidence': confidence,
            'q_values': q_values.tolist(),
            'model_type': 'RL_DQN'
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'Hold': probabilities[0],
                'Buy': probabilities[1], 
                'Sell': probabilities[2]
            }
        
        return result
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            data: Test data
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        data_clean = self.prepare_data(data)
        
        # Create evaluation environment
        eval_env = TradingEnvironment(
            data=data_clean,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
            lookback_window=self.lookback_window
        )
        
        # Run evaluation episode
        state = eval_env.reset()
        total_reward = 0
        actions_taken = []
        
        while True:
            action = self.agent.act(state, training=False)
            next_state, reward, done, info = eval_env.step(action)
            
            actions_taken.append(action)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Get performance metrics
        performance = eval_env.get_portfolio_performance()
        
        # Calculate additional metrics
        action_distribution = np.bincount(actions_taken, minlength=3)
        action_percentages = action_distribution / len(actions_taken) * 100
        
        metrics = {
            'total_reward': total_reward,
            'final_portfolio_value': performance.get('final_value', self.initial_balance),
            'total_return': performance.get('total_return', 0.0),
            'max_drawdown': performance.get('max_drawdown', 0.0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0.0),
            'num_trades': performance.get('num_trades', 0),
            'hold_percentage': action_percentages[0],
            'buy_percentage': action_percentages[1],
            'sell_percentage': action_percentages[2]
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained RL model."""
        if not self.is_trained:
            logger.warning("Saving untrained model")
        
        # Save agent
        self.agent.save(filepath)
        
        # Save metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        self.model_info['training_history'] = self.training_history
        self.model_info['is_trained'] = self.is_trained
        
        with open(metadata_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)
        
        logger.info(f"RL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained RL model."""
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_info = json.load(f)
                self.training_history = self.model_info.get('training_history', [])
                self.is_trained = self.model_info.get('is_trained', False)
        
        # Load the agent data to get state_size
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
            
            if 'state_size' in agent_data and 'action_size' in agent_data:
                # Initialize agent with correct dimensions
                from src.rl_dqn_agent import DQNAgent
                self.agent = DQNAgent(
                    state_size=agent_data['state_size'],
                    action_size=agent_data['action_size'],
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    epsilon=self.epsilon,
                    epsilon_min=self.epsilon_min,
                    epsilon_decay=self.epsilon_decay
                )
                
                # Load the saved weights/model
                self.agent.load(filepath)
                logger.info(f"RL model and agent loaded from {filepath}")
            else:
                logger.warning(f"Invalid model file format: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            self.agent = None
            self.is_trained = False
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = "models/rl_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"rl_checkpoint_episode_{episode}.pkl")
        self.agent.save(checkpoint_path)
        
        logger.info(f"Checkpoint saved at episode {episode}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for display."""
        info = self.model_info.copy()
        
        if self.agent:
            info['agent_stats'] = self.agent.get_training_stats()
        
        if self.training_history:
            latest_training = self.training_history[-1]
            info['latest_performance'] = {
                'episodes_trained': latest_training.get('episodes_trained', 0),
                'total_return': latest_training.get('total_return', 0.0),
                'best_portfolio_value': latest_training.get('best_portfolio_value', 0.0)
            }
        
        return info


def test_rl_model():
    """Test the RL model integration."""
    print("Testing RL Model Integration...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 50000,
        'high': np.random.randn(200).cumsum() + 50000,
        'low': np.random.randn(200).cumsum() + 50000,
        'close': np.random.randn(200).cumsum() + 50000,
        'volume': np.random.rand(200) * 1000,
        'rsi': np.random.rand(200) * 100,
        'macd': np.random.randn(200),
        'bb_upper': np.random.randn(200).cumsum() + 50000,
        'bb_lower': np.random.randn(200).cumsum() + 50000,
    }, index=dates)
    
    # Initialize model
    rl_model = RLTradingModel()
    
    # Train on first 150 samples
    train_data = data.head(150)
    metrics = rl_model.train(train_data, episodes=20)
    print(f"Training metrics: {metrics}")
    
    # Evaluate on last 50 samples
    test_data = data.tail(70)  # Need overlap for lookback window
    eval_metrics = rl_model.evaluate(test_data)
    print(f"Evaluation metrics: {eval_metrics}")
    
    # Test prediction
    recent_data = data.tail(30)
    prediction = rl_model.predict(recent_data, return_probabilities=True)
    print(f"Prediction: {prediction}")
    
    # Test save/load
    rl_model.save_model("test_rl_model.pkl")
    print("Model saved successfully")
    
    print("RL Model Integration test completed!")


if __name__ == "__main__":
    test_rl_model()