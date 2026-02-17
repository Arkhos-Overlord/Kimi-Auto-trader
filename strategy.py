from ml_model import MLModel
import pandas as pd
import logging

class TradingStrategy:
    def __init__(self, exchange):
        self.exchange = exchange
        # Initialize model with the processed NSE data
        self.ml_model = MLModel(data_path="data.csv")
        self.ml_model.train()
        # Mapping for IndMoney Security IDs
        self.security_mapping = {
            'NIFTY50': 'NSE_INDEX_NIFTY50', # Note: Index trading might differ from stock trading
            'RELIANCE': '2885'
        }

    def execute_strategy(self):
        logging.info("Executing trading strategy with ML insights...")
        
        # In a real scenario, you'd fetch the latest OHLC data and calculate these features
        # For demonstration, we use dummy current features matching the model's expected columns
        feature_cols = ['return', 'volatility', 'rsi', 'ema9', 'ema21', 'ema_signal']
        current_features = pd.DataFrame([[50, 100, 45, 24000, 23800, 1]], columns=feature_cols)
        
        prediction = self.ml_model.predict(current_features)

        if prediction == 1:
            logging.info("SIGNAL: BUY/LONG for RELIANCE")
            # Example: Placing a small test order on IndMoney
            # self.exchange.place_order(security_id=self.security_mapping['RELIANCE'], side='BUY', qty=1)
        else:
            logging.info("SIGNAL: SELL/SHORT/WAIT")
            # self.exchange.place_order(security_id=self.security_mapping['RELIANCE'], side='SELL', qty=1)
