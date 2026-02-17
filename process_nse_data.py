import pandas as pd

def process_data(input_file, output_file):
    print(f"Processing {input_file}...")
    # Read the CSV, skipping the header rows that yfinance adds
    try:
        df = pd.read_csv(input_file, skiprows=2)
        # Rename columns for clarity based on observation
        # Format: Date, Open, High, Low, Close, Volume
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert to numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Feature Engineering
        # Feature 1: Daily Return (Close - Open)
        # Feature 2: Volatility (High - Low)
        # Feature 3: Volume Change (Percentage)
        df['feature1'] = df['Close'] - df['Open']
        df['feature2'] = df['High'] - df['Low']
        
        # Target: 1 if next day's Close is higher than today's Close, else 0
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop the last row as it won't have a target
        df = df.dropna()
        
        # Save processed data for ML
        df[['feature1', 'feature2', 'target']].to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing data: {e}")
        return False

if __name__ == "__main__":
    process_data("nifty50_data.csv", "data.csv")
