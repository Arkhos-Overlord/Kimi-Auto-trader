from ml_model import MLModel
import pandas as pd

class TradingStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        # Initialize model with the processed NSE data
        self.ml_model = MLModel(data_path="data.csv")
        self.ml_model.train()

    def execute_strategy(self):
        print("\n--- Executing Indian Market Strategy with ML Insights ---")
        
        # In a real scenario, you'd fetch the latest OHLC data from the exchange
        # For demonstration, let's assume current daily return is +50 and volatility is 100
        # (These values are relative to the Nifty 50 index points)
        current_features = pd.DataFrame([[50, 100]], columns=['feature1', 'feature2'])
        
        prediction = self.ml_model.predict(current_features)

        if prediction == 1:
            print("SIGNAL: ML Model predicts UPWARD movement for the next session. [BUY/LONG]")
        else:
            print("SIGNAL: ML Model predicts DOWNWARD/NEUTRAL movement. [SELL/SHORT/WAIT]")
        
        print("--------------------------------------------------------\n")
