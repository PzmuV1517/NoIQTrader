"""
Reinforcement Learning Training Script

Train the RL trading agent on historical Bitcoin data and save the trained model.
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
from datetime import datetime
import json

# Import RL components
from src.rl_model import RLTradingModel
from src.data_loader import BTCDataLoader
from src.feature_engineer import FeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str = 'data/btc_featured_data.csv') -> pd.DataFrame:
    """
    Load and prepare data for RL training.
    
    Args:
        data_path: Path to the featured data CSV file
        
    Returns:
        Prepared DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        # Try to load existing featured data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded existing featured data: {data.shape}")
        
    except FileNotFoundError:
        logger.info("Featured data not found. Creating from raw data...")
        
        # Load raw data and create features
        loader = BTCDataLoader()
        raw_data = loader.load_data(save_to_file=True)
        
        # Engineer features
        engineer = FeatureEngineer()
        data = engineer.create_all_features(raw_data)
        
        # Save featured data
        os.makedirs('data', exist_ok=True)
        data.to_csv(data_path)
        logger.info(f"Created and saved featured data: {data.shape}")
    
    # Ensure we have enough data
    if len(data) < 100:
        raise ValueError(f"Insufficient data for training. Got {len(data)} samples, need at least 100")
    
    logger.info(f"Data preparation complete. Shape: {data.shape}")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    return data


def train_rl_model(
    data: pd.DataFrame,
    episodes: int = 200,
    train_split: float = 0.8,
    lookback_window: int = 20,
    learning_rate: float = 0.001,
    initial_balance: float = 10000.0,
    transaction_cost: float = 0.001,
    save_path: str = 'models/rl_trading_model.pkl'
) -> RLTradingModel:
    """
    Train the RL trading model.
    
    Args:
        data: Training data
        episodes: Number of training episodes
        train_split: Training/validation split ratio
        lookback_window: Number of historical observations for state
        learning_rate: Learning rate for DQN
        initial_balance: Starting portfolio balance
        transaction_cost: Transaction fee percentage
        save_path: Path to save the trained model
        
    Returns:
        Trained RL model
    """
    logger.info("Initializing RL trading model...")
    
    # Initialize RL model
    rl_model = RLTradingModel(
        lookback_window=lookback_window,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        learning_rate=learning_rate
    )
    
    # Split data
    split_idx = int(len(data) * train_split)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx - lookback_window:]  # Include overlap for lookback
    
    logger.info(f"Training on {len(train_data)} samples")
    logger.info(f"Validation on {len(val_data)} samples")
    
    # Train model
    logger.info(f"Starting training for {episodes} episodes...")
    training_metrics = rl_model.train(
        data=train_data,
        episodes=episodes,
        save_freq=max(1, episodes // 10),  # Save checkpoints 10 times during training
        early_stopping=True,
        patience=20
    )
    
    logger.info("Training completed!")
    logger.info(f"Training metrics: {training_metrics}")
    
    # Evaluate on validation data
    logger.info("Evaluating on validation data...")
    val_metrics = rl_model.evaluate(val_data)
    logger.info(f"Validation metrics: {val_metrics}")
    
    # Save trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rl_model.save_model(save_path)
    
    # Save training results
    results = {
        'training_metrics': training_metrics,
        'validation_metrics': val_metrics,
        'model_parameters': {
            'episodes': episodes,
            'train_split': train_split,
            'lookback_window': lookback_window,
            'learning_rate': learning_rate,
            'initial_balance': initial_balance,
            'transaction_cost': transaction_cost
        },
        'data_info': {
            'total_samples': len(data),
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'features': list(data.columns)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = save_path.replace('.pkl', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training results saved to {results_path}")
    
    return rl_model


def test_trained_model(model: RLTradingModel, test_data: pd.DataFrame):
    """
    Test the trained model and display results.
    
    Args:
        model: Trained RL model
        test_data: Test data
    """
    logger.info("Testing trained model...")
    
    # Make predictions on recent data
    recent_data = test_data.tail(50)  # Use last 50 days
    
    try:
        prediction = model.predict(recent_data, return_probabilities=True)
        
        logger.info("=== RL Model Prediction ===")
        logger.info(f"Recommended Action: {prediction['action_name']}")
        logger.info(f"Confidence: {prediction['confidence']:.2%}")
        logger.info(f"Q-values: {prediction['q_values']}")
        
        if 'probabilities' in prediction:
            probs = prediction['probabilities']
            logger.info("Action Probabilities:")
            logger.info(f"  Hold: {probs['Hold']:.2%}")
            logger.info(f"  Buy:  {probs['Buy']:.2%}")
            logger.info(f"  Sell: {probs['Sell']:.2%}")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
    
    # Get model info
    model_info = model.get_model_info()
    logger.info("=== Model Information ===")
    logger.info(f"Model Type: {model_info['model_type']}")
    logger.info(f"Is Trained: {model_info.get('is_trained', False)}")
    
    if 'latest_performance' in model_info:
        perf = model_info['latest_performance']
        logger.info(f"Episodes Trained: {perf['episodes_trained']}")
        logger.info(f"Total Return: {perf['total_return']:.2%}")
        logger.info(f"Best Portfolio Value: ${perf['best_portfolio_value']:.2f}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train RL Trading Model')
    parser.add_argument('--data-path', default='data/btc_featured_data.csv',
                       help='Path to featured data CSV file')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for DQN')
    parser.add_argument('--lookback-window', type=int, default=20,
                       help='Number of historical observations for state')
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                       help='Starting portfolio balance')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction fee percentage')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training/validation split ratio')
    parser.add_argument('--save-path', default='models/rl_trading_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing model without training')
    
    args = parser.parse_args()
    
    logger.info("=== RL Trading Model Training ===")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load and prepare data
        data = load_and_prepare_data(args.data_path)
        
        if args.test_only:
            # Load existing model and test
            logger.info("Loading existing model for testing...")
            rl_model = RLTradingModel()
            rl_model.load_model(args.save_path)
            test_trained_model(rl_model, data)
            
        else:
            # Train new model
            rl_model = train_rl_model(
                data=data,
                episodes=args.episodes,
                train_split=args.train_split,
                lookback_window=args.lookback_window,
                learning_rate=args.learning_rate,
                initial_balance=args.initial_balance,
                transaction_cost=args.transaction_cost,
                save_path=args.save_path
            )
            
            # Test the trained model
            test_trained_model(rl_model, data)
        
        logger.info("RL model training/testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()