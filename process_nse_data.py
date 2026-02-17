import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_data(input_file, output_file):
    print(f"Processing {input_file} with advanced features...")
    try:
        df = pd.read_csv(input_file, skiprows=2)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # --- Feature Engineering ---
        # 1. Daily Return
        df['return'] = df['Close'] - df['Open']
        
        # 2. Volatility (High-Low spread)
        df['volatility'] = df['High'] - df['Low']
        
        # 3. RSI (Relative Strength Index)
        df['rsi'] = calculate_rsi(df['Close'])
        
        # 4. EMA (Exponential Moving Average) - 9 and 21 periods
        df['ema9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # 5. EMA Crossover Signal (1 if EMA9 > EMA21, else 0)
        df['ema_signal'] = (df['ema9'] > df['ema21']).astype(int)
        
        # Target: 1 if next day's Close is higher than today's Close, else 0
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop rows with NaN values (from RSI and shift)
        df = df.dropna()
        
        # Save processed data for ML
        features = ['return', 'volatility', 'rsi', 'ema9', 'ema21', 'ema_signal', 'target']
        df[features].to_csv(output_file, index=False)
        print(f"Advanced features saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing data: {e}")
        return False

if __name__ == "__main__":
    process_data("nifty50_data.csv", "data.csv")
