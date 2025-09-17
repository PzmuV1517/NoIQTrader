"""
Model Persistence and Loading System for NoIQTrader

This module handles saving and loading trained ML models to avoid retraining
on each application run.
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the original ML models
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_models import TradingSignalPredictor


class ModelManager:
    """
    Manages saving, loading, and prediction with trained ML models.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store saved models
        """
        self.models_dir = models_dir
        self.create_models_directory()
        
        # Model information
        self.model_info = {}
        self.predictor = None
        
    def create_models_directory(self):
        """Create models directory if it doesn't exist."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"📁 Created models directory: {self.models_dir}")
    
    def save_models(self, predictor: TradingSignalPredictor) -> Dict[str, str]:
        """
        Save all trained models and metadata to disk.
        
        Args:
            predictor: Trained TradingSignalPredictor instance
            
        Returns:
            Dictionary with paths to saved files
        """
        if not predictor.models:
            raise ValueError("No trained models found in predictor")
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("💾 Saving trained models...")
        
        # Save individual models
        for model_name, model_info in predictor.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
            
            # Save the model using joblib (better for sklearn models)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model_info['model'],
                    'scaler': model_info.get('scaler'),
                    'metrics': model_info['metrics'],
                    'predictions': {
                        'train': model_info['predictions']['train'].tolist(),
                        'test': model_info['predictions']['test'].tolist()
                    }
                }, f)
            
            saved_files[model_name] = model_path
            print(f"  ✅ Saved {model_name} to {model_path}")
        
        # Save predictor metadata
        metadata = {
            'feature_names': predictor.feature_names,
            'target_threshold': getattr(predictor, 'target_threshold', 0.01),
            'train_size': len(predictor.X_train) if hasattr(predictor, 'X_train') else 0,
            'test_size': len(predictor.X_test) if hasattr(predictor, 'X_test') else 0,
            'data_shape': predictor.data.shape if hasattr(predictor, 'data') else (0, 0),
            'saved_timestamp': timestamp,
            'model_files': saved_files
        }
        
        metadata_path = os.path.join(self.models_dir, f"model_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        saved_files['metadata'] = metadata_path
        
        # Save latest models (without timestamp for easy loading)
        latest_dir = os.path.join(self.models_dir, "latest")
        if not os.path.exists(latest_dir):
            os.makedirs(latest_dir)
        
        for model_name, model_path in saved_files.items():
            if model_name != 'metadata':
                latest_path = os.path.join(latest_dir, f"{model_name}.pkl")
                import shutil
                shutil.copy2(model_path, latest_path)
                print(f"  📋 Copied {model_name} to latest/")
        
        # Copy metadata to latest
        latest_metadata_path = os.path.join(latest_dir, "model_metadata.json")
        import shutil
        shutil.copy2(metadata_path, latest_metadata_path)
        
        print(f"\n✅ All models saved successfully!")
        print(f"📊 Models directory: {self.models_dir}")
        print(f"🔄 Latest models: {latest_dir}")
        
        return saved_files
    
    def load_latest_models(self, data_path: str = None) -> TradingSignalPredictor:
        """
        Load the latest saved models.
        
        Args:
            data_path: Path to data file (optional, for re-initialization)
            
        Returns:
            TradingSignalPredictor instance with loaded models
        """
        latest_dir = os.path.join(self.models_dir, "latest")
        metadata_path = os.path.join(latest_dir, "model_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No latest models found. Please train models first.")
        
        print("📂 Loading latest models...")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)
        
        # Initialize predictor
        if data_path:
            self.predictor = TradingSignalPredictor(data_path)
            self.predictor.load_data()
        else:
            self.predictor = TradingSignalPredictor()
        
        # Restore metadata
        self.predictor.feature_names = metadata['feature_names']
        
        # Load individual models
        loaded_models = {}
        for model_name in ['logistic_regression', 'random_forest', 'lstm']:
            model_file = os.path.join(latest_dir, f"{model_name}.pkl")
            
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                loaded_models[model_name] = {
                    'model': model_data['model'],
                    'scaler': model_data.get('scaler'),
                    'metrics': model_data['metrics'],
                    'predictions': {
                        'train': np.array(model_data['predictions']['train']),
                        'test': np.array(model_data['predictions']['test'])
                    }
                }
                print(f"  ✅ Loaded {model_name}")
        
        self.predictor.models = loaded_models
        
        print(f"✅ Loaded {len(loaded_models)} models successfully!")
        print(f"📊 Available models: {list(loaded_models.keys())}")
        
        return self.predictor
    
    def predict_with_model(self, model_name: str, features: np.ndarray) -> Dict:
        """
        Make prediction with a specific model.
        
        Args:
            model_name: Name of the model to use
            features: Feature array for prediction
            
        Returns:
            Dictionary with prediction and probabilities
        """
        if not self.predictor or model_name not in self.predictor.models:
            raise ValueError(f"Model {model_name} not available")
        
        model_info = self.predictor.models[model_name]
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        # Scale features if scaler exists
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probabilities if available
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            class_names = ['Sell', 'Hold', 'Buy']
            for i, class_name in enumerate(class_names):
                probabilities[class_name] = proba[i]
        
        return {
            'prediction': int(prediction),
            'probabilities': probabilities,
            'model_used': model_name
        }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        if not self.predictor:
            return {}
        
        info = {
            'models_available': list(self.predictor.models.keys()),
            'feature_count': len(self.predictor.feature_names) if self.predictor.feature_names else 0,
            'features': self.predictor.feature_names
        }
        
        # Add model performance
        for model_name, model_info in self.predictor.models.items():
            info[f'{model_name}_metrics'] = model_info['metrics']
        
        return info


def create_predictions_dataset(data_path: str, models_dir: str = "models", 
                             output_path: str = "data/btc_with_predictions.csv") -> str:
    """
    Create a dataset with model predictions for backtesting.
    
    Args:
        data_path: Path to featured data
        models_dir: Directory containing saved models
        output_path: Path to save predictions dataset
        
    Returns:
        Path to the created predictions dataset
    """
    print("🔮 Creating predictions dataset for backtesting...")
    
    # Load models
    manager = ModelManager(models_dir)
    try:
        predictor = manager.load_latest_models(data_path)
    except FileNotFoundError:
        print("❌ No saved models found. Training new models...")
        # Train models if none exist
        predictor = TradingSignalPredictor(data_path)
        predictor.load_data()
        
        # Prepare features
        X, y = predictor.prepare_features()
        predictor.split_data(X, y, test_size=0.2)
        
        # Train models
        predictor.train_logistic_regression()
        predictor.train_random_forest()
        
        # Save models
        manager.save_models(predictor)
    
    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Create target variable
    data_with_target = predictor.create_target_variable()
    
    # Prepare features for the entire dataset
    if not predictor.feature_names:
        # If feature names not available, recreate them
        X_temp, _ = predictor.prepare_features()
        feature_columns = predictor.feature_names
    else:
        feature_columns = predictor.feature_names
    
    X_full = data_with_target[feature_columns].fillna(data_with_target[feature_columns].median())
    
    # Make predictions with best model (Random Forest)
    if 'random_forest' in predictor.models:
        model_info = predictor.models['random_forest']
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        if scaler:
            X_scaled = scaler.transform(X_full)
        else:
            X_scaled = X_full
        
        # Get predictions and probabilities
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Add predictions to data
        data_with_target['model_prediction'] = predictions
        data_with_target['sell_prob'] = probabilities[:, 0]  # Sell probability
        data_with_target['hold_prob'] = probabilities[:, 1]  # Hold probability  
        data_with_target['buy_prob'] = probabilities[:, 2]   # Buy probability
        data_with_target['confidence'] = np.max(probabilities, axis=1)  # Highest probability
        
        print(f"✅ Added predictions from Random Forest model")
        
        # Print prediction distribution
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"Prediction distribution:")
        for signal, count in pred_counts.items():
            signal_name = ['Sell', 'Hold', 'Buy'][signal + 1]
            print(f"  {signal_name}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # Save the dataset with predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_with_target.to_csv(output_path)
    
    print(f"💾 Saved predictions dataset to: {output_path}")
    print(f"📊 Dataset shape: {data_with_target.shape}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Test model persistence
    print("🧪 Testing Model Persistence System...")
    
    data_path = "data/btc_featured_data.csv"
    
    # Create predictions dataset
    predictions_path = create_predictions_dataset(data_path)
    
    # Test loading models
    manager = ModelManager()
    try:
        predictor = manager.load_latest_models(data_path)
        print(f"\n📋 Model Info:")
        info = manager.get_model_info()
        for key, value in info.items():
            if not isinstance(value, list):
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
