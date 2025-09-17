#!/usr/bin/env python3
"""
Complete Model Training Script
Trains all NoIQTrader models: Random Forest, Logistic Regression, and RL
"""

import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_ml_models():
    """Train traditional ML models (Random Forest & Logistic Regression)"""
    logger.info("=== Training ML Models ===")
    
    try:
        from src.ml_models import TradingSignalPredictor
        
        # Initialize predictor
        predictor = TradingSignalPredictor()
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        predictor.load_data()
        predictor.create_target_variable()
        
        # Prepare features
        X, y = predictor.prepare_features()
        predictor.split_data(X, y)
        
        # Train models
        logger.info("Training Logistic Regression...")
        lr_results = predictor.train_logistic_regression()
        
        logger.info("Training Random Forest...")
        rf_results = predictor.train_random_forest()
        
        # Save models
        from src.model_persistence import ModelManager
        manager = ModelManager()
        manager.save_models(predictor)
        
        # Results summary
        logger.info("=== ML Training Results ===")
        logger.info(f"Logistic Regression - Test Accuracy: {lr_results['metrics']['test_accuracy']:.3f}")
        logger.info(f"Random Forest - Test Accuracy: {rf_results['metrics']['test_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        return False

def train_rl_model(episodes=100, learning_rate=0.001):
    """Train Reinforcement Learning model"""
    logger.info("=== Training RL Model ===")
    
    try:
        from src.rl_model import RLTradingModel
        from src.data_loader import BTCDataLoader
        from src.feature_engineer import FeatureEngineer
        import pandas as pd
        
        # Load data
        logger.info("Loading data for RL training...")
        data = pd.read_csv('data/btc_featured_data.csv', index_col=0, parse_dates=True)
        
        # Split data
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]
        
        # Initialize RL model
        rl_model = RLTradingModel(
            lookback_window=20,
            initial_balance=10000.0,
            transaction_cost=0.001
        )
        
        # Train model
        logger.info(f"Training RL model for {episodes} episodes...")
        training_metrics = rl_model.train(
            train_data,
            episodes=episodes,
            early_stopping=False
        )
        
        # Evaluate on validation data
        logger.info("Evaluating RL model...")
        eval_metrics = rl_model.evaluate(val_data)
        
        # Save model
        rl_model.save_model('models/rl_trading_model.pkl')
        
        # Results summary
        logger.info("=== RL Training Results ===")
        logger.info(f"Episodes Trained: {training_metrics['episodes_trained']}")
        logger.info(f"Final Portfolio Value: ${training_metrics['final_portfolio_value']:,.2f}")
        logger.info(f"Total Return: {training_metrics['total_return']:.2%}")
        logger.info(f"Validation Return: {eval_metrics['total_return']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train all NoIQTrader models')
    parser.add_argument('--ml-only', action='store_true', help='Train only ML models')
    parser.add_argument('--rl-only', action='store_true', help='Train only RL model')
    parser.add_argument('--episodes', type=int, default=100, help='RL training episodes')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='RL learning rate')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"=== NoIQTrader Model Training Started ===")
    logger.info(f"Start time: {start_time}")
    
    success = True
    
    if not args.rl_only:
        logger.info("Training ML models...")
        ml_success = train_ml_models()
        success = success and ml_success
        
    if not args.ml_only:
        logger.info("Training RL model...")
        rl_success = train_rl_model(args.episodes, args.learning_rate)
        success = success and rl_success
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"=== Training Completed ===")
    logger.info(f"Duration: {duration}")
    logger.info(f"Success: {success}")
    
    if success:
        logger.info("All models trained successfully!")
        logger.info("You can now run: streamlit run app.py")
    else:
        logger.error("Some models failed to train. Check logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()