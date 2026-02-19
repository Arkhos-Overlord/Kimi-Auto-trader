"""
Daily Data Updater Module
Implements rolling window approach for continuous model learning
- Fetches new market data daily
- Maintains 2-year rolling window (494 trading days)
- Triggers weekly retraining
- Logs all updates for monitoring
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('data_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DailyDataUpdater:
    """Manages daily data updates with rolling window approach"""
    
    def __init__(self, data_file='nifty50_2years.csv', window_days=494):
        """
        Initialize the data updater
        
        Args:
            data_file: Path to the CSV file containing historical data
            window_days: Number of trading days to maintain (default: 494 = 2 years)
        """
        self.data_file = data_file
        self.window_days = window_days
        self.log_file = 'data_update_log.json'
        self.last_update_date = None
        
    def fetch_latest_data(self, ticker='^NSEI', days=5):
        """
        Fetch latest market data from yfinance
        
        Args:
            ticker: Stock ticker (default: ^NSEI = Nifty 50)
            days: Number of days to fetch (default: 5 for buffer)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching latest data for {ticker}...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data fetched for {ticker}")
                return None
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'Date'}, inplace=True)
            
            # Format date as string
            data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"Successfully fetched {len(data)} rows of data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
    
    def load_existing_data(self):
        """Load existing data from CSV file"""
        try:
            if not os.path.exists(self.data_file):
                logger.warning(f"Data file {self.data_file} not found")
                return None
            
            df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(df)} rows from {self.data_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def merge_new_data(self, existing_df, new_df):
        """
        Merge new data with existing data, removing duplicates
        
        Args:
            existing_df: Existing data
            new_df: New data to add
            
        Returns:
            Merged DataFrame
        """
        if existing_df is None:
            return new_df
        
        if new_df is None:
            return existing_df
        
        try:
            # Combine dataframes
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates, keeping the latest
            combined['Date'] = pd.to_datetime(combined['Date'])
            combined = combined.drop_duplicates(subset=['Date'], keep='last')
            combined = combined.sort_values('Date').reset_index(drop=True)
            
            # Convert date back to string
            combined['Date'] = combined['Date'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"Merged data: {len(combined)} total rows")
            return combined
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return existing_df
    
    def apply_rolling_window(self, df):
        """
        Apply rolling window to keep only last N trading days
        
        Args:
            df: DataFrame to trim
            
        Returns:
            Trimmed DataFrame with rolling window applied
        """
        if len(df) <= self.window_days:
            logger.info(f"Data size ({len(df)}) within window ({self.window_days})")
            return df
        
        # Keep only the last window_days rows
        trimmed = df.tail(self.window_days).reset_index(drop=True)
        removed_rows = len(df) - len(trimmed)
        
        logger.info(f"Applied rolling window: removed {removed_rows} old rows")
        logger.info(f"Current data window: {trimmed['Date'].iloc[0]} to {trimmed['Date'].iloc[-1]}")
        
        return trimmed
    
    def save_data(self, df):
        """Save data to CSV file"""
        try:
            df.to_csv(self.data_file, index=False)
            logger.info(f"Saved {len(df)} rows to {self.data_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def log_update_event(self, event_type, details):
        """Log update events for monitoring"""
        try:
            import json
            
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'details': details
            }
            
            # Append to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
            
            logger.info(f"Logged event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error logging event: {str(e)}")
    
    def update_daily(self):
        """
        Main daily update function
        - Fetches latest data
        - Merges with existing data
        - Applies rolling window
        - Saves updated data
        - Logs the update
        
        Returns:
            True if update successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info("STARTING DAILY DATA UPDATE")
        logger.info("=" * 70)
        
        try:
            # Check if already updated today
            today = datetime.now().strftime('%Y-%m-%d')
            if self.last_update_date == today:
                logger.info("Already updated today, skipping...")
                return False
            
            # Fetch new data
            new_data = self.fetch_latest_data()
            if new_data is None or new_data.empty:
                logger.warning("No new data fetched")
                return False
            
            # Load existing data
            existing_data = self.load_existing_data()
            
            # Merge data
            merged_data = self.merge_new_data(existing_data, new_data)
            
            # Apply rolling window
            windowed_data = self.apply_rolling_window(merged_data)
            
            # Save updated data
            if self.save_data(windowed_data):
                self.last_update_date = today
                
                # Log the update
                self.log_update_event('daily_update', {
                    'rows_added': len(new_data),
                    'total_rows': len(windowed_data),
                    'date_range': f"{windowed_data['Date'].iloc[0]} to {windowed_data['Date'].iloc[-1]}"
                })
                
                logger.info("✓ Daily update completed successfully")
                logger.info("=" * 70)
                return True
            else:
                logger.error("Failed to save data")
                return False
                
        except Exception as e:
            logger.error(f"Error in daily update: {str(e)}")
            self.log_update_event('update_error', {'error': str(e)})
            return False
    
    def get_update_statistics(self):
        """Get statistics about data updates"""
        try:
            df = self.load_existing_data()
            if df is None:
                return None
            
            stats = {
                'total_rows': len(df),
                'date_range': f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}",
                'trading_days': len(df),
                'years_of_data': len(df) / 252,
                'window_size': self.window_days,
                'at_capacity': len(df) >= self.window_days,
                'last_update': self.last_update_date
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    updater = DailyDataUpdater()
    
    # Get current statistics
    stats = updater.get_update_statistics()
    if stats:
        print("\n📊 CURRENT DATA STATISTICS")
        print("-" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Perform daily update
    print("\n🔄 PERFORMING DAILY UPDATE")
    print("-" * 50)
    success = updater.update_daily()
    
    if success:
        # Get updated statistics
        updated_stats = updater.get_update_statistics()
        print("\n✓ UPDATE COMPLETED")
        print("-" * 50)
        for key, value in updated_stats.items():
            print(f"{key}: {value}")
    else:
        print("\n✗ UPDATE FAILED")
