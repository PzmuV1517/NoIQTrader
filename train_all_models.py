#!/usr/bin/env python3
"""
Interactive Model Training Menu for NoIQTrader
Select models, configure parameters, and queue training jobs
"""

import sys
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingQueue:
    """Manages a queue of training jobs with configurations"""
    
    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self.completed_jobs: List[Dict[str, Any]] = []
    
    def add_job(self, job_config: Dict[str, Any]):
        """Add a training job to the queue"""
        job_config['id'] = len(self.queue) + 1
        job_config['status'] = 'queued'
        job_config['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.queue.append(job_config)
        print(f" Added job #{job_config['id']}: {job_config['model_type']} to queue")
    
    def remove_job(self, job_id: int):
        """Remove a job from the queue"""
        self.queue = [job for job in self.queue if job['id'] != job_id]
        print(f"  Removed job #{job_id} from queue")
    
    def clear_queue(self):
        """Clear all jobs from the queue"""
        self.queue.clear()
        print(" Queue cleared")
    
    def show_queue(self):
        """Display current queue"""
        if not self.queue:
            print(" Queue is empty")
            return
        
        print("\n Current Training Queue:")
        print("=" * 80)
        for i, job in enumerate(self.queue):
            print(f"#{job['id']} | {job['model_type']} | Status: {job['status']}")
            if job['model_type'] == 'ML Models':
                print(f"    Models: {', '.join(job['models'])}")
            elif job['model_type'] == 'RL Model':
                print(f"    Episodes: {job['episodes']} | Learning Rate: {job['learning_rate']}")
                print(f"    Initial Balance: ${job['initial_balance']:,} | Transaction Cost: {job['transaction_cost']}")
            print(f"    Created: {job['created_at']}")
            print("-" * 80)

class ModelTrainer:
    """Handles actual model training execution"""
    
    def __init__(self):
        self.training_results = []
    
    def train_ml_models(self, models_to_train: List[str]) -> bool:
        """Train selected ML models"""
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
            
            results = {}
            
            # Train selected models
            if 'logistic_regression' in models_to_train:
                logger.info("Training Logistic Regression...")
                lr_results = predictor.train_logistic_regression()
                results['logistic_regression'] = lr_results
            
            if 'random_forest' in models_to_train:
                logger.info("Training Random Forest...")
                rf_results = predictor.train_random_forest()
                results['random_forest'] = rf_results
            
            # Save models
            from src.model_persistence import ModelManager
            manager = ModelManager()
            manager.save_models(predictor)
            
            # Results summary
            logger.info("=== ML Training Results ===")
            for model_name, model_results in results.items():
                logger.info(f"{model_name.replace('_', ' ').title()} - Test Accuracy: {model_results['metrics']['test_accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            return False
    
    def train_rl_model(self, config: Dict[str, Any]) -> bool:
        """Train RL model with given configuration"""
        logger.info("=== Training RL Model ===")
        
        try:
            from src.rl_model import RLTradingModel
            import pandas as pd
            
            # Load data
            logger.info("Loading data for RL training...")
            data = pd.read_csv('data/btc_featured_data.csv', index_col=0, parse_dates=True)
            
            # Split data
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]
            
            # Initialize RL model with configuration
            rl_model = RLTradingModel(
                lookback_window=config.get('lookback_window', 20),
                initial_balance=config.get('initial_balance', 10000.0),
                transaction_cost=config.get('transaction_cost', 0.001),
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            # Train model
            logger.info(f"Training RL model for {config['episodes']} episodes...")
            training_metrics = rl_model.train(
                train_data,
                episodes=config['episodes'],
                early_stopping=config.get('early_stopping', False)
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

class TrainingMenu:
    """Interactive menu system for model training"""
    
    def __init__(self):
        self.queue = TrainingQueue()
        self.trainer = ModelTrainer()
        self.running = True
    
    def show_main_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print(" NoIQTrader - Interactive Model Training Menu")
        print("="*60)
        print("1. Configure ML Model Training")
        print("2. Configure RL Model Training") 
        print("3. View Training Queue")
        print("4.  Manage Queue (Remove/Clear)")
        print("5. Start Training Queue")
        print("6. View Previous Training Results")
        print("7.  Advanced Configuration")
        print("8. Refresh Data Files")
        print("9. Exit")
        print("="*60)
    
    def configure_ml_training(self):
        """Configure ML model training parameters"""
        print("\n ML Model Training Configuration")
        print("-" * 40)
        
        # Select models to train
        print("Select models to train:")
        print("1. Logistic Regression")
        print("2. Random Forest")
        print("3. Both Models")
        
        while True:
            choice = input("\nChoice (1-3): ").strip()
            if choice == '1':
                models = ['logistic_regression']
                break
            elif choice == '2':
                models = ['random_forest'] 
                break
            elif choice == '3':
                models = ['logistic_regression', 'random_forest']
                break
            else:
                print(" Invalid choice. Please enter 1, 2, or 3.")
        
        # Additional configuration
        print(f"\nSelected models: {', '.join([m.replace('_', ' ').title() for m in models])}")
        
        # Create job configuration
        job_config = {
            'model_type': 'ML Models',
            'models': models,
            'data_split': 0.8,  # Default train/test split
            'cross_validation': True
        }
        
        # Add to queue
        self.queue.add_job(job_config)
        
        input("\n ML training job configured. Press Enter to continue...")
    
    def configure_rl_training(self):
        """Configure RL model training parameters"""
        print("\n RL Model Training Configuration")
        print("-" * 40)
        
        # Episodes configuration
        print("Training Episodes:")
        print("1. Quick Test (100 episodes) - ~5 minutes")
        print("2. Short Training (500 episodes) - ~20 minutes") 
        print("3. Medium Training (1000 episodes) - ~40 minutes")
        print("4. Long Training (3000 episodes) - ~2 hours")
        print("5. Extended Training (5000 episodes) - ~4 hours")
        print("6. Custom episodes")
        
        while True:
            choice = input("\nChoice (1-6): ").strip()
            if choice == '1':
                episodes = 100
                break
            elif choice == '2':
                episodes = 500
                break
            elif choice == '3':
                episodes = 1000
                break
            elif choice == '4':
                episodes = 3000
                break
            elif choice == '5':
                episodes = 5000
                break
            elif choice == '6':
                while True:
                    try:
                        episodes = int(input("Enter number of episodes: "))
                        if episodes > 0:
                            break
                        else:
                            print(" Episodes must be positive")
                    except ValueError:
                        print(" Please enter a valid number")
                break
            else:
                print(" Invalid choice. Please enter 1-6.")
        
        # Learning rate
        print(f"\nSelected episodes: {episodes}")
        print("\nLearning Rate:")
        print("1. Conservative (0.0001) - Stable but slow")
        print("2. Standard (0.001) - Balanced") 
        print("3. Aggressive (0.01) - Fast but potentially unstable")
        print("4. Custom learning rate")
        
        while True:
            choice = input("\nChoice (1-4): ").strip()
            if choice == '1':
                learning_rate = 0.0001
                break
            elif choice == '2':
                learning_rate = 0.001
                break
            elif choice == '3':
                learning_rate = 0.01
                break
            elif choice == '4':
                while True:
                    try:
                        learning_rate = float(input("Enter learning rate (e.g., 0.001): "))
                        if 0 < learning_rate < 1:
                            break
                        else:
                            print(" Learning rate should be between 0 and 1")
                    except ValueError:
                        print(" Please enter a valid number")
                break
            else:
                print(" Invalid choice. Please enter 1-4.")
        
        # Initial balance
        print(f"\nLearning rate: {learning_rate}")
        while True:
            try:
                initial_balance = float(input("\nInitial balance (default 10000): ") or "10000")
                if initial_balance > 0:
                    break
                else:
                    print(" Initial balance must be positive")
            except ValueError:
                print(" Please enter a valid number")
        
        # Transaction cost
        print(f"Initial balance: ${initial_balance:,.2f}")
        while True:
            try:
                transaction_cost = float(input("\nTransaction cost % (default 0.1): ") or "0.1") / 100
                if 0 <= transaction_cost <= 0.1:
                    break
                else:
                    print(" Transaction cost should be between 0 and 10%")
            except ValueError:
                print(" Please enter a valid number")
        
        # Early stopping
        print(f"Transaction cost: {transaction_cost*100:.1f}%")
        early_stopping = input("\nEnable early stopping? (y/N): ").strip().lower() == 'y'
        
        # Create job configuration
        job_config = {
            'model_type': 'RL Model',
            'episodes': episodes,
            'learning_rate': learning_rate,
            'initial_balance': initial_balance,
            'transaction_cost': transaction_cost,
            'early_stopping': early_stopping,
            'lookback_window': 20  # Default
        }
        
        # Add to queue
        self.queue.add_job(job_config)
        
        input("\n RL training job configured. Press Enter to continue...")
    
    def manage_queue(self):
        """Manage training queue"""
        while True:
            print("\n  Queue Management")
            print("-" * 30)
            self.queue.show_queue()
            
            if not self.queue.queue:
                input("\nPress Enter to return to main menu...")
                return
            
            print("\n1. Remove specific job")
            print("2. Clear entire queue")
            print("3. Return to main menu")
            
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == '1':
                try:
                    job_id = int(input("Enter job ID to remove: "))
                    self.queue.remove_job(job_id)
                except ValueError:
                    print(" Please enter a valid job ID")
            elif choice == '2':
                confirm = input("  Clear entire queue? (y/N): ").strip().lower()
                if confirm == 'y':
                    self.queue.clear_queue()
            elif choice == '3':
                return
            else:
                print(" Invalid choice")
    
    def start_training(self):
        """Execute all jobs in the training queue"""
        if not self.queue.queue:
            print(" Queue is empty. Add some training jobs first!")
            input("Press Enter to continue...")
            return
        
        print("\n Starting Training Queue")
        print("=" * 50)
        
        # Show queue summary
        total_jobs = len(self.queue.queue)
        ml_jobs = len([j for j in self.queue.queue if j['model_type'] == 'ML Models'])
        rl_jobs = len([j for j in self.queue.queue if j['model_type'] == 'RL Model'])
        
        print(f"Total jobs: {total_jobs}")
        print(f"ML jobs: {ml_jobs}")
        print(f"RL jobs: {rl_jobs}")
        
        # Estimate time
        total_episodes = sum([j.get('episodes', 0) for j in self.queue.queue if j['model_type'] == 'RL Model'])
        estimated_time = (ml_jobs * 5) + (total_episodes * 0.05)  # Rough estimate
        print(f"Estimated time: ~{estimated_time:.0f} minutes")
        
        confirm = input(f"\n  Start training {total_jobs} jobs? (y/N): ").strip().lower()
        if confirm != 'y':
            print(" Training cancelled")
            return
        
        # Execute jobs
        start_time = datetime.now()
        successful_jobs = 0
        
        for i, job in enumerate(self.queue.queue):
            print(f"\n{'='*60}")
            print(f" Executing Job #{job['id']} ({i+1}/{total_jobs})")
            print(f"Model Type: {job['model_type']}")
            print(f"{'='*60}")
            
            job['status'] = 'running'
            job['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            success = False
            
            try:
                if job['model_type'] == 'ML Models':
                    success = self.trainer.train_ml_models(job['models'])
                elif job['model_type'] == 'RL Model':
                    success = self.trainer.train_rl_model(job)
                
                if success:
                    job['status'] = 'completed'
                    successful_jobs += 1
                    print(f" Job #{job['id']} completed successfully")
                else:
                    job['status'] = 'failed'
                    print(f" Job #{job['id']} failed")
                
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = str(e)
                logger.error(f"Job #{job['id']} failed with error: {e}")
                print(f" Job #{job['id']} failed with error: {e}")
            
            job['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.queue.completed_jobs.append(job)
        
        # Clear queue after execution
        self.queue.queue.clear()
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(" Training Queue Completed!")
        print(f"{'='*60}")
        print(f"Total jobs: {total_jobs}")
        print(f"Successful: {successful_jobs}")
        print(f"Failed: {total_jobs - successful_jobs}")
        print(f"Duration: {duration}")
        print(f"{'='*60}")
        
        if successful_jobs == total_jobs:
            print(" All models trained successfully!")
            print(" You can now run: streamlit run app.py")
        else:
            print("  Some training jobs failed. Check logs above.")
        
        input("\nPress Enter to continue...")
    
    def view_results(self):
        """View previous training results"""
        print("\n Previous Training Results")
        print("-" * 40)
        
        if not self.queue.completed_jobs:
            print("No completed training jobs found.")
            input("Press Enter to continue...")
            return
        
        for job in self.queue.completed_jobs:
            print(f"\nJob #{job['id']} - {job['model_type']}")
            print(f"Status: {job['status']}")
            print(f"Started: {job.get('started_at', 'N/A')}")
            print(f"Completed: {job.get('completed_at', 'N/A')}")
            if job['status'] == 'failed' and 'error' in job:
                print(f"Error: {job['error']}")
            print("-" * 30)
        
        input("\nPress Enter to continue...")
    
    def advanced_config(self):
        """Advanced configuration options"""
        print("\n  Advanced Configuration")
        print("-" * 30)
        print("1. Check data files")
        print("2. View system info")
        print("3. Export queue configuration")
        print("4. Import queue configuration")
        print("5. Clean model cache")
        print("6. Return to main menu")
        
        choice = input("\nChoice (1-6): ").strip()
        
        if choice == '1':
            self.check_data_files()
        elif choice == '2':
            self.show_system_info()
        elif choice == '3':
            self.export_queue_config()
        elif choice == '4':
            self.import_queue_config()
        elif choice == '5':
            self.clean_model_cache()
        elif choice == '6':
            return
        else:
            print(" Invalid choice")
    
    def check_data_files(self):
        """Check if required data files exist"""
        required_files = [
            'data/btc_featured_data.csv',
            'data/btc_with_predictions.csv'
        ]
        
        print("\n Data File Status:")
        all_present = True
        
        for file_path in required_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f" {file_path} ({size:.1f} MB)")
            else:
                print(f" {file_path} (Missing)")
                all_present = False
        
        if not all_present:
            print("\n  Some data files are missing!")
            print(" Run: python process_data.py to generate them")
        
        input("\nPress Enter to continue...")
    
    def show_system_info(self):
        """Show system information"""
        print("\n System Information:")
        print(f"Python version: {sys.version}")
        print(f"Working directory: {os.getcwd()}")
        
        # Check for required modules
        modules = ['pandas', 'numpy', 'sklearn', 'tensorflow', 'streamlit']
        print("\n Module Status:")
        
        for module in modules:
            try:
                __import__(module)
                print(f" {module}")
            except ImportError:
                print(f" {module} (Not installed)")
        
        input("\nPress Enter to continue...")
    
    def export_queue_config(self):
        """Export current queue to JSON file"""
        if not self.queue.queue:
            print(" Queue is empty, nothing to export")
            input("Press Enter to continue...")
            return
        
        filename = f"queue_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.queue.queue, f, indent=2)
            print(f" Queue configuration exported to: {filename}")
        except Exception as e:
            print(f" Export failed: {e}")
        
        input("Press Enter to continue...")
    
    def import_queue_config(self):
        """Import queue configuration from JSON file"""
        filename = input("Enter JSON filename to import: ").strip()
        
        if not os.path.exists(filename):
            print(f" File not found: {filename}")
            input("Press Enter to continue...")
            return
        
        try:
            with open(filename, 'r') as f:
                imported_queue = json.load(f)
            
            self.queue.queue.extend(imported_queue)
            print(f" Imported {len(imported_queue)} jobs from {filename}")
        except Exception as e:
            print(f" Import failed: {e}")
        
        input("Press Enter to continue...")
    
    def clean_model_cache(self):
        """Clean model cache and temporary files"""
        cache_dirs = ['models/__pycache__', 'src/__pycache__', '__pycache__']
        
        confirm = input("  Clean model cache? (y/N): ").strip().lower()
        if confirm != 'y':
            return
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                print(f" Cleaned: {cache_dir}")
        
        print(" Cache cleaned")
        input("Press Enter to continue...")
    
    def refresh_data(self):
        """Refresh data files"""
        print("\n Refreshing Data Files")
        print("-" * 30)
        
        confirm = input("  This will regenerate data files. Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            return
        
        try:
            print(" Running data processing...")
            os.system("python process_data.py")
            print(" Data refresh completed")
        except Exception as e:
            print(f" Data refresh failed: {e}")
        
        input("Press Enter to continue...")
    
    def run(self):
        """Run the interactive menu"""
        print(" Welcome to NoIQTrader Interactive Training!")
        
        while self.running:
            try:
                self.show_main_menu()
                choice = input("\nEnter your choice (1-9): ").strip()
                
                if choice == '1':
                    self.configure_ml_training()
                elif choice == '2':
                    self.configure_rl_training()
                elif choice == '3':
                    self.queue.show_queue()
                    input("\nPress Enter to continue...")
                elif choice == '4':
                    self.manage_queue()
                elif choice == '5':
                    self.start_training()
                elif choice == '6':
                    self.view_results()
                elif choice == '7':
                    self.advanced_config()
                elif choice == '8':
                    self.refresh_data()
                elif choice == '9':
                    print(" Goodbye!")
                    self.running = False
                else:
                    print(" Invalid choice. Please enter 1-9.")
                    input("Press Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n An error occurred: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create and run menu
    menu = TrainingMenu()
    menu.run()

if __name__ == "__main__":
    main()