"""
ML Models Module for NoIQTrader
Implements machine learning models for Bitcoin trading signal prediction
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy nn module for when torch is not available
    class DummyNN:
        class Module:
            pass
        class LSTM:
            pass
        class Linear:
            pass
        class CrossEntropyLoss:
            pass
    if not TORCH_AVAILABLE:
        nn = DummyNN()

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSignalPredictor:
    """
    Machine learning predictor for Bitcoin trading signals
    """
    
    def __init__(self, data_path: str = "data/btc_featured_data.csv"):
        """
        Initialize the trading signal predictor
        
        Args:
            data_path (str): Path to the featured dataset
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        
        logger.info("Initializing Trading Signal Predictor")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the featured dataset
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        logger.info(f"Loading data from {self.data_path}")
        
        self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        
        logger.info(f"Loaded {len(self.data)} rows with {len(self.data.columns)} features")
        logger.info(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
        return self.data
    
    def create_target_variable(self, return_threshold: float = 0.01) -> pd.DataFrame:
        """
        Create trading signal target variable based on next-day returns
        
        Args:
            return_threshold (float): Threshold for Buy/Sell signals (default 1%)
            
        Returns:
            pd.DataFrame: Data with target variable added
        """
        logger.info(f"Creating target variable with threshold: {return_threshold*100:.1f}%")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate next day return
        self.data['next_day_return'] = self.data['Close'].pct_change().shift(-1)
        
        # Create trading signals
        conditions = [
            self.data['next_day_return'] > return_threshold,    # Buy signal
            self.data['next_day_return'] < -return_threshold,   # Sell signal
        ]
        choices = [1, -1]  # Buy=1, Sell=-1
        
        self.data['target'] = np.select(conditions, choices, default=0)  # Hold=0
        
        # Remove rows with missing target (last row)
        self.data = self.data.dropna(subset=['target'])
        
        # Print target distribution
        target_counts = self.data['target'].value_counts().sort_index()
        logger.info("Target distribution:")
        logger.info(f"  Sell (-1): {target_counts.get(-1, 0)} ({target_counts.get(-1, 0)/len(self.data)*100:.1f}%)")
        logger.info(f"  Hold (0):  {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.data)*100:.1f}%)")
        logger.info(f"  Buy (1):   {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.data)*100:.1f}%)")
        
        return self.data
    
    def prepare_features(self, exclude_features: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning
        
        Args:
            exclude_features (list): Features to exclude from modeling
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and target (y)
        """
        logger.info("Preparing features for modeling")
        
        if exclude_features is None:
            exclude_features = [
                'target', 'next_day_return', 
                'Open', 'High', 'Low', 'Close', 'Volume',  # Original OHLCV
                'Dividends', 'Stock Splits'  # Not useful for crypto
            ]
        
        # Select features (exclude target and specified columns)
        feature_cols = [col for col in self.data.columns if col not in exclude_features]
        
        # Remove any columns with all NaN values
        feature_cols = [col for col in feature_cols if not self.data[col].isnull().all()]
        
        self.feature_names = feature_cols
        logger.info(f"Using {len(feature_cols)} features for modeling")
        
        # Prepare X and y
        X = self.data[feature_cols].values
        y = self.data['target'].values
        
        # Handle any remaining NaN values
        if np.isnan(X).any():
            logger.warning("Found NaN values in features, filling with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> None:
        """
        Split data into train/test sets with time-series consideration
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            test_size (float): Proportion of test set
        """
        logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        # For time series, we split by time (not random)
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale features
        logger.info("Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Print class distribution in train/test
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        logger.info("Train set distribution:")
        for class_val in [-1, 0, 1]:
            count = train_dist.get(class_val, 0)
            logger.info(f"  Class {class_val}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        logger.info("Test set distribution:")
        for class_val in [-1, 0, 1]:
            count = test_dist.get(class_val, 0)
            logger.info(f"  Class {class_val}: {count} ({count/len(self.y_test)*100:.1f}%)")
    
    def train_logistic_regression(self, **kwargs) -> Dict[str, Any]:
        """
        Train Logistic Regression model
        
        Returns:
            Dict[str, Any]: Model and metrics
        """
        logger.info("Training Logistic Regression model...")
        
        # Default parameters
        params = {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'  # Handle class imbalance
        }
        params.update(kwargs)
        
        # Train model
        lr_model = LogisticRegression(**params)
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred_train = lr_model.predict(self.X_train_scaled)
        y_pred_test = lr_model.predict(self.X_test_scaled)
        y_pred_proba_test = lr_model.predict_proba(self.X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            self.y_train, y_pred_train, self.y_test, y_pred_test, y_pred_proba_test
        )
        
        # Store model
        self.models['logistic_regression'] = {
            'model': lr_model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'test_proba': y_pred_proba_test
            }
        }
        
        logger.info(f"Logistic Regression - Test Accuracy: {metrics['test_accuracy']:.3f}")
        
        return self.models['logistic_regression']
    
    def train_random_forest(self, **kwargs) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Returns:
            Dict[str, Any]: Model and metrics
        """
        logger.info("Training Random Forest model...")
        
        # Default parameters
        params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        params.update(kwargs)
        
        # Train model
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(self.X_train, self.y_train)  # RF doesn't need scaling
        
        # Predictions
        y_pred_train = rf_model.predict(self.X_train)
        y_pred_test = rf_model.predict(self.X_test)
        y_pred_proba_test = rf_model.predict_proba(self.X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            self.y_train, y_pred_train, self.y_test, y_pred_test, y_pred_proba_test
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store model
        self.models['random_forest'] = {
            'model': rf_model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'test_proba': y_pred_proba_test
            }
        }
        
        logger.info(f"Random Forest - Test Accuracy: {metrics['test_accuracy']:.3f}")
        
        return self.models['random_forest']
    
    def train_lstm(self, sequence_length: int = 30, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Train LSTM model (optional, requires PyTorch)
        
        Args:
            sequence_length (int): Length of input sequences
            
        Returns:
            Dict[str, Any] or None: Model and metrics if successful
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping LSTM model.")
            return None
        
        logger.info("Training LSTM model...")
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = self._create_sequences(
            self.X_train_scaled, self.y_train, sequence_length
        )
        X_test_seq, y_test_seq = self._create_sequences(
            self.X_test_scaled, self.y_test, sequence_length
        )
        
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            logger.warning("Not enough data for LSTM sequences. Skipping LSTM model.")
            return None
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.LongTensor(y_train_seq + 1)  # Convert -1,0,1 to 0,1,2
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.LongTensor(y_test_seq + 1)
        
        # Create LSTM model
        input_size = X_train_seq.shape[2]
        hidden_size = kwargs.get('hidden_size', 64)
        num_layers = kwargs.get('num_layers', 2)
        
        lstm_model = LSTMModel(input_size, hidden_size, num_layers, 3)  # 3 classes
        
        # Train LSTM
        lstm_model = self._train_lstm_model(
            lstm_model, X_train_tensor, y_train_tensor, 
            X_test_tensor, y_test_tensor, **kwargs
        )
        
        # Get predictions
        with torch.no_grad():
            lstm_model.eval()
            y_pred_proba_test = torch.softmax(lstm_model(X_test_tensor), dim=1).numpy()
            y_pred_test = np.argmax(y_pred_proba_test, axis=1) - 1  # Convert back to -1,0,1
            
            y_pred_proba_train = torch.softmax(lstm_model(X_train_tensor), dim=1).numpy()
            y_pred_train = np.argmax(y_pred_proba_train, axis=1) - 1
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_train_seq, y_pred_train, y_test_seq, y_pred_test, y_pred_proba_test
        )
        
        # Store model
        self.models['lstm'] = {
            'model': lstm_model,
            'metrics': metrics,
            'sequence_length': sequence_length,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'test_proba': y_pred_proba_test
            }
        }
        
        logger.info(f"LSTM - Test Accuracy: {metrics['test_accuracy']:.3f}")
        
        return self.models['lstm']
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _train_lstm_model(self, model, X_train, y_train, X_test, y_test, **kwargs):
        """Train the LSTM model"""
        epochs = kwargs.get('epochs', 50)
        learning_rate = kwargs.get('learning_rate', 0.001)
        batch_size = kwargs.get('batch_size', 32)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        return model
    
    def _calculate_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred, y_test_proba) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation"""
        
        # Basic metrics
        train_accuracy = accuracy_score(y_train_true, y_train_pred)
        test_accuracy = accuracy_score(y_test_true, y_test_pred)
        
        # Classification metrics (macro average for multiclass)
        precision = precision_score(y_test_true, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_true, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models
        
        Returns:
            pd.DataFrame: Model comparison table
        """
        logger.info("Comparing model performances...")
        
        comparison_data = []
        
        for model_name, model_info in self.models.items():
            metrics = model_info['metrics']
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        # Sort by test accuracy
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot confusion matrices for all models"""
        
        n_models = len(self.models)
        if n_models == 0:
            logger.warning("No models trained yet.")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        class_names = ['Sell', 'Hold', 'Buy']
        
        for idx, (model_name, model_info) in enumerate(self.models.items()):
            y_pred_test = model_info['predictions']['test']
            
            # For LSTM, we need to adjust the test target
            if model_name == 'lstm':
                y_test = model_info['predictions']['test']  # Already adjusted in LSTM training
                cm = confusion_matrix(y_test, y_pred_test)
            else:
                cm = confusion_matrix(self.y_test, y_pred_test)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def predict_next_action(self, model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Predict next trading action using the latest data
        
        Args:
            model_name (str): Name of the model to use for prediction
            
        Returns:
            Dict[str, Any]: Prediction results with confidence
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        logger.info(f"Making next-day prediction using {model_name} model...")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Get the latest features (last row)
        latest_features = self.data[self.feature_names].iloc[-1:].values
        
        # Handle any NaN values
        if np.isnan(latest_features).any():
            logger.warning("NaN values in latest features, filling with median")
            latest_features = np.nan_to_num(latest_features, nan=np.nanmedian(latest_features))
        
        # Scale features if needed
        if model_name == 'logistic_regression':
            latest_features_scaled = self.scaler.transform(latest_features)
            prediction = model.predict(latest_features_scaled)[0]
            probabilities = model.predict_proba(latest_features_scaled)[0]
        elif model_name == 'random_forest':
            prediction = model.predict(latest_features)[0]
            probabilities = model.predict_proba(latest_features)[0]
        elif model_name == 'lstm':
            # For LSTM, we need a sequence
            sequence_length = model_info['sequence_length']
            if len(self.data) < sequence_length:
                raise ValueError(f"Not enough data for LSTM prediction. Need {sequence_length} samples.")
            
            latest_sequence = self.data[self.feature_names].iloc[-sequence_length:].values
            latest_sequence = self.scaler.transform(latest_sequence)
            latest_sequence = torch.FloatTensor(latest_sequence).unsqueeze(0)
            
            with torch.no_grad():
                model.eval()
                output = model(latest_sequence)
                probabilities = torch.softmax(output, dim=1).numpy()[0]
                prediction = np.argmax(probabilities) - 1  # Convert back to -1,0,1
        
        # Map prediction to action
        action_map = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}
        action = action_map[prediction]
        
        # Get confidence (max probability)
        if model_name == 'lstm':
            confidence = float(np.max(probabilities))
            prob_dict = {
                'Sell': float(probabilities[0]),
                'Hold': float(probabilities[1]),  
                'Buy': float(probabilities[2])
            }
        else:
            confidence = float(np.max(probabilities))
            # Map probabilities to actions (order: Sell, Hold, Buy)
            unique_classes = model.classes_
            prob_dict = {}
            for i, class_val in enumerate(unique_classes):
                prob_dict[action_map[class_val]] = float(probabilities[i])
        
        # Current BTC info
        current_price = self.data['Close'].iloc[-1]
        current_date = self.data.index[-1].date()
        
        result = {
            'date': current_date,
            'current_price': current_price,
            'prediction': prediction,
            'action': action,
            'confidence': confidence,
            'probabilities': prob_dict,
            'model_used': model_name
        }
        
        logger.info(f"Prediction: {action} (confidence: {confidence:.3f})")
        
        return result


class LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """LSTM model for trading signal prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        if TORCH_AVAILABLE:
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            raise ImportError("PyTorch not available for LSTM model")
        
    def forward(self, x):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for LSTM model")
            
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last output
        return out


def main():
    """Main function for testing the trading predictor"""
    
    # Initialize predictor
    predictor = TradingSignalPredictor()
    
    # Load and prepare data
    predictor.load_data()
    predictor.create_target_variable()
    
    # Prepare features
    X, y = predictor.prepare_features()
    predictor.split_data(X, y)
    
    # Train models
    print("\n🤖 Training Models...")
    predictor.train_logistic_regression()
    predictor.train_random_forest()
    
    # Try LSTM if PyTorch is available
    if TORCH_AVAILABLE:
        predictor.train_lstm()
    
    # Compare models
    print("\n📊 Model Comparison:")
    comparison = predictor.compare_models()
    print(comparison)
    
    # Make prediction
    print("\n🎯 Next-Day Prediction:")
    best_model = comparison.iloc[0]['Model'].lower().replace(' ', '_')
    prediction = predictor.predict_next_action(best_model)
    
    print(f"Date: {prediction['date']}")
    print(f"Current BTC Price: ${prediction['current_price']:,.2f}")
    print(f"Recommended Action: {prediction['action']}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Model Used: {prediction['model_used']}")


if __name__ == "__main__":
    main()
