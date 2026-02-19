"""
Weekly Retraining Scheduler
Automatically retrains the ensemble model every week with fresh data
- Monitors retraining schedule
- Triggers model retraining
- Tracks model performance improvements
- Logs all retraining events
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from sklearn.model_selection import train_test_split
from enhanced_ml_model import EnhancedMLModel
from enhanced_features import EnhancedFeatureEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyRetrainer:
    """Manages weekly model retraining with performance tracking"""
    
    def __init__(self, data_file='nifty50_2years.csv', model_dir='models'):
        """
        Initialize the retrainer
        
        Args:
            data_file: Path to the data CSV file
            model_dir: Directory to save trained models
        """
        self.data_file = data_file
        self.model_dir = model_dir
        self.retraining_log = 'retraining_log.json'
        self.last_retrain_date = None
        self.model_performance_history = []
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load retraining history
        self._load_history()
    
    def _load_history(self):
        """Load previous retraining history"""
        try:
            if os.path.exists(self.retraining_log):
                with open(self.retraining_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Get the last entry
                        last_entry = json.loads(lines[-1])
                        self.last_retrain_date = last_entry.get('timestamp')
                        logger.info(f"Loaded retraining history: {len(lines)} entries")
        except Exception as e:
            logger.warning(f"Could not load history: {str(e)}")
    
    def should_retrain(self):
        """
        Check if retraining should be performed
        
        Returns:
            Tuple (should_retrain: bool, reason: str)
        """
        today = datetime.now()
        
        # First retrain: if no history
        if self.last_retrain_date is None:
            return True, "First retraining (no history)"
        
        # Parse last retrain date
        try:
            last_retrain = datetime.fromisoformat(self.last_retrain_date)
            days_since = (today - last_retrain).days
            
            # Retrain if 7 days have passed
            if days_since >= 7:
                return True, f"Weekly schedule ({days_since} days since last retrain)"
            else:
                return False, f"Not yet (only {days_since} days since last retrain)"
                
        except Exception as e:
            logger.warning(f"Error parsing date: {str(e)}")
            return True, "Date parsing error, forcing retrain"
    
    def load_and_prepare_data(self):
        """Load and prepare data for retraining"""
        try:
            logger.info("Loading data...")
            df = pd.read_csv(self.data_file)
            
            if df.empty:
                logger.error("Data file is empty")
                return None, None, None
            
            logger.info(f"Loaded {len(df)} rows of data")
            
            # Engineer features
            logger.info("Engineering features...")
            feature_engineer = EnhancedFeatureEngine()
            df_features = feature_engineer.engineer_features(df)
            
            # Handle missing values
            df_features = df_features.dropna()
            logger.info(f"After feature engineering: {len(df_features)} rows")
            
            if len(df_features) < 100:
                logger.error("Insufficient data after feature engineering")
                return None, None, None
            
            # Create target variable (1 if next day positive, 0 otherwise)
            df_features['target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
            df_features = df_features.dropna()
            
            # Prepare X and y
            feature_cols = [col for col in df_features.columns 
                          if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']]
            X = df_features[feature_cols].values
            y = df_features['target'].values
            
            logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
            
            return (X_train, y_train), (X_test, y_test), feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None, None
    
    def retrain_model(self):
        """
        Perform model retraining
        
        Returns:
            Dictionary with retraining results
        """
        logger.info("=" * 70)
        logger.info("STARTING WEEKLY MODEL RETRAINING")
        logger.info("=" * 70)
        
        try:
            # Check if should retrain
            should_retrain, reason = self.should_retrain()
            logger.info(f"Retrain check: {reason}")
            
            if not should_retrain:
                return {
                    'status': 'skipped',
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Load and prepare data
            train_data, test_data, feature_cols = self.load_and_prepare_data()
            
            if train_data is None:
                logger.error("Failed to prepare data")
                return {
                    'status': 'failed',
                    'reason': 'Data preparation failed',
                    'timestamp': datetime.now().isoformat()
                }
            
            X_train, y_train = train_data
            X_test, y_test = test_data
            
            # Train model
            logger.info("Training ensemble model...")
            model = EnhancedMLModel()
            model.train(X_train, y_train)
            
            # Evaluate model
            logger.info("Evaluating model...")
            train_accuracy = model.evaluate(X_train, y_train)
            test_accuracy = model.evaluate(X_test, y_test)
            
            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            
            # Save model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(self.model_dir, f'model_{timestamp}.pkl')
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Prepare results
            result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(feature_cols),
                'model_path': model_path,
                'improvement': None
            }
            
            # Calculate improvement
            if self.model_performance_history:
                last_accuracy = self.model_performance_history[-1]['test_accuracy']
                improvement = test_accuracy - last_accuracy
                result['improvement'] = float(improvement)
                logger.info(f"Accuracy improvement: {improvement:+.4f}")
            
            # Store in history
            self.model_performance_history.append(result)
            self.last_retrain_date = result['timestamp']
            
            # Log the result
            self._log_retraining_event(result)
            
            logger.info("✓ Retraining completed successfully")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
            error_result = {
                'status': 'failed',
                'reason': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self._log_retraining_event(error_result)
            return error_result
    
    def _log_retraining_event(self, result):
        """Log retraining event to file"""
        try:
            with open(self.retraining_log, 'a') as f:
                f.write(json.dumps(result) + '\n')
            logger.info("Logged retraining event")
        except Exception as e:
            logger.error(f"Error logging event: {str(e)}")
    
    def get_retraining_history(self, limit=10):
        """Get recent retraining history"""
        try:
            history = []
            if os.path.exists(self.retraining_log):
                with open(self.retraining_log, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        history.append(json.loads(line))
            return history
        except Exception as e:
            logger.error(f"Error reading history: {str(e)}")
            return []
    
    def get_performance_summary(self):
        """Get summary of model performance over time"""
        try:
            history = self.get_retraining_history(limit=100)
            
            if not history:
                return None
            
            successful = [h for h in history if h.get('status') == 'success']
            
            if not successful:
                return None
            
            accuracies = [h.get('test_accuracy', 0) for h in successful]
            
            summary = {
                'total_retrainings': len(successful),
                'avg_accuracy': np.mean(accuracies),
                'max_accuracy': np.max(accuracies),
                'min_accuracy': np.min(accuracies),
                'latest_accuracy': accuracies[-1] if accuracies else None,
                'accuracy_trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'stable'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    retrainer = WeeklyRetrainer()
    
    # Check if should retrain
    should_retrain, reason = retrainer.should_retrain()
    print(f"\n📅 RETRAIN CHECK")
    print("-" * 50)
    print(f"Should retrain: {should_retrain}")
    print(f"Reason: {reason}")
    
    # Get performance summary
    summary = retrainer.get_performance_summary()
    if summary:
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 50)
        for key, value in summary.items():
            print(f"{key}: {value}")
    
    # Perform retraining
    print(f"\n🔄 RETRAINING MODEL")
    print("-" * 50)
    result = retrainer.retrain_model()
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Test Accuracy: {result.get('test_accuracy'):.4f}")
        if result.get('improvement'):
            print(f"Improvement: {result.get('improvement'):+.4f}")
