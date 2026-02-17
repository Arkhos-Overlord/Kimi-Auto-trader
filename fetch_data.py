import requests
import pandas as pd
from datetime import datetime

def fetch_binance_data(symbol="BTCUSDT", interval="1d", limit=100):
    print(f"Fetching {limit} days of historical data for {symbol}...")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Binance kline data format:
        # [
        #   [
        #     1499040000000,      // Open time
        #     "0.01634790",       // Open
        #     "0.80000000",       // High
        #     "0.01575800",       // Low
        #     "0.01577100",       // Close
        #     "148976.11427815",  // Volume
        #     1499644799999,      // Close time
        #     "2434.19055334",    // Quote asset volume
        #     308,                // Number of trades
        #     "1756.87402397",    // Taker buy base asset volume
        #     "28.46694368",      // Taker buy quote asset volume
        #     "17928899.62484339" // Ignore
        #   ]
        # ]
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Create features for ML
        # Feature 1: Price change (Close - Open)
        # Feature 2: High - Low spread
        # Target: 1 if next day Close > current day Close, else 0
        df['feature1'] = df['close'] - df['open']
        df['feature2'] = df['high'] - df['low']
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop last row as it won't have a target
        df = df.dropna()
        
        # Save to CSV
        df[['feature1', 'feature2', 'target']].to_csv("data.csv", index=False)
        print("Data saved to data.csv")
        return True
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False

if __name__ == "__main__":
    fetch_binance_data()
