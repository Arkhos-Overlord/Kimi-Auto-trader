"""
Enhanced Feature Engineering Module for Kimi-Auto-trader
Implements advanced technical indicators for improved ML predictions
"""

import pandas as pd
import numpy as np
from typing import Tuple

class EnhancedFeatureEngine:
    """Advanced technical indicator calculations"""
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent_smooth.rolling(window=smooth_d).mean()
        
        return k_percent_smooth, d_percent
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume-Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        return vwap
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span).mean()
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Momentum indicator"""
        return prices - prices.shift(period)
    
    @staticmethod
    def calculate_rate_of_change(prices: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change"""
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (plus_dm.rolling(window=period).mean() / atr)
        di_minus = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicators"""
        df = df.copy()
        
        # Basic price features
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'].diff()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Trend indicators
        df['SMA_10'] = EnhancedFeatureEngine.calculate_sma(df['Close'], 10)
        df['SMA_20'] = EnhancedFeatureEngine.calculate_sma(df['Close'], 20)
        df['SMA_50'] = EnhancedFeatureEngine.calculate_sma(df['Close'], 50)
        df['EMA_12'] = EnhancedFeatureEngine.calculate_ema(df['Close'], 12)
        df['EMA_26'] = EnhancedFeatureEngine.calculate_ema(df['Close'], 26)
        df['EMA_Crossover'] = (df['EMA_12'] > df['EMA_26']).astype(int)
        
        # Momentum indicators
        df['RSI_14'] = EnhancedFeatureEngine.calculate_rsi(df['Close'], 14)
        df['Momentum'] = EnhancedFeatureEngine.calculate_momentum(df['Close'], 10)
        df['ROC'] = EnhancedFeatureEngine.calculate_rate_of_change(df['Close'], 12)
        
        # MACD
        macd_line, signal_line, histogram = EnhancedFeatureEngine.calculate_macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = histogram
        df['MACD_Cross'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        # Stochastic
        k_percent, d_percent = EnhancedFeatureEngine.calculate_stochastic(df['High'], df['Low'], df['Close'])
        df['Stochastic_K'] = k_percent
        df['Stochastic_D'] = d_percent
        df['Stochastic_Cross'] = (df['Stochastic_K'] > df['Stochastic_D']).astype(int)
        
        # Bollinger Bands
        upper_band, middle_band, lower_band = EnhancedFeatureEngine.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = upper_band
        df['BB_Middle'] = middle_band
        df['BB_Lower'] = lower_band
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volatility indicators
        df['Volatility'] = df['Daily_Return'].rolling(window=5).std()
        df['ATR'] = EnhancedFeatureEngine.calculate_atr(df['High'], df['Low'], df['Close'])
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Volume indicators
        df['VWAP'] = EnhancedFeatureEngine.calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = EnhancedFeatureEngine.calculate_obv(df['Close'], df['Volume'])
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Additional indicators
        df['Williams_R'] = EnhancedFeatureEngine.calculate_williams_r(df['High'], df['Low'], df['Close'])
        df['ADX'] = EnhancedFeatureEngine.calculate_adx(df['High'], df['Low'], df['Close'])
        df['CCI'] = EnhancedFeatureEngine.calculate_cci(df['High'], df['Low'], df['Close'])
        
        # Price position indicators
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Open_Close_Ratio'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Volatility-adjusted indicators
        df['RSI_Volatility'] = df['RSI_14'] * df['Volatility']
        df['Momentum_Volatility'] = df['Momentum'] * df['Volatility']
        
        return df
