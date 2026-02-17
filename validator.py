import logging
import pandas as pd
import numpy as np

class AccuracyValidator:
    def __init__(self, model, threshold=0.70):
        self.model = model
        self.threshold = threshold
        self.performance_history = []
        self.recent_accuracy = 1.0

    def validate_recent_performance(self):
        """
        Simulates validation of recent predictions against actual market outcomes.
        In production, this would compare the bot's 'Buy/Sell' signals 
        with what actually happened in the market after the signal.
        """
        logging.info("Validating recent model accuracy...")
        
        # Simulated validation logic:
        # We take a small sample of recent data and check accuracy
        try:
            df = pd.read_csv("data.csv")
            sample = df.tail(10)
            X = sample[['return', 'volatility', 'rsi', 'ema9', 'ema21', 'ema_signal']]
            y_true = sample['target']
            
            # Use model to predict on this recent sample
            y_pred = [self.model.predict(pd.DataFrame([X.iloc[i]], columns=X.columns)) for i in range(len(X))]
            
            # Calculate accuracy
            correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
            self.recent_accuracy = correct / len(y_true)
            
            self.performance_history.append(self.recent_accuracy)
            logging.info(f"Recent Validation Accuracy: {self.recent_accuracy:.2%}")
            
        except Exception as e:
            logging.error(f"Validation error: {e}")

    def needs_retraining(self):
        """Returns True if accuracy falls below the threshold."""
        return self.recent_accuracy < self.threshold

    def reset_metrics(self):
        """Resets history after re-training."""
        self.performance_history = []
        logging.info("Validation metrics reset after re-training.")
