import pandas as pd
from ml_model import MLModel

class Backtester:
    def __init__(self, data_path="data.csv"):
        self.data_path = data_path
        self.ml_model = MLModel(data_path=data_path)
        self.ml_model.train()

    def run_backtest(self):
        print("\n--- Starting Backtest Simulation ---")
        df = pd.read_csv(self.data_path)
        
        # We'll simulate on the last 20% of the data (the 'test' portion)
        test_size = int(len(df) * 0.2)
        test_df = df.tail(test_size).copy()
        
        initial_balance = 100000  # ₹1 Lakh
        balance = initial_balance
        position = 0
        trades = 0
        successful_trades = 0

        feature_cols = ['return', 'volatility', 'rsi', 'ema9', 'ema21', 'ema_signal']

        for i in range(len(test_df)):
            current_row = test_df.iloc[i]
            features = pd.DataFrame([current_row[feature_cols].values], columns=feature_cols)
            
            prediction = self.ml_model.predict(features)
            actual_outcome = current_row['target']

            # Trading Logic: 
            # If prediction is 1 (UP), we "buy" a fixed amount.
            # If prediction is 0 (DOWN), we "sell" or stay out.
            if prediction == 1:
                trades += 1
                if actual_outcome == 1:
                    successful_trades += 1
                    balance += 1000  # Simulated profit per successful trade
                else:
                    balance -= 1000  # Simulated loss per unsuccessful trade

        print(f"Initial Balance: ₹{initial_balance}")
        print(f"Final Balance: ₹{balance}")
        print(f"Total Trades: {trades}")
        print(f"Win Rate: {(successful_trades/trades*100 if trades > 0 else 0):.2f}%")
        print("--- Backtest Completed ---\n")

if __name__ == "__main__":
    tester = Backtester()
    tester.run_backtest()
