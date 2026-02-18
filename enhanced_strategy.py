"""
Enhanced Trading Strategy with Risk Management and Position Sizing
Implements Kelly Criterion, Dynamic Stop-Loss, and Pyramid Entry
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class EnhancedTradingStrategy:
    """Advanced trading strategy with risk management"""
    
    def __init__(self, initial_balance: float = 100000, max_risk_per_trade: float = 0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.trades = []
        self.win_rate = 0.5
        self.avg_win = 1.0
        self.avg_loss = 1.0
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        f* = (bp - q) / b where b = odds, p = win probability, q = loss probability
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.1
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        f = (b * p - q) / b
        
        # Cap at 25% for safety
        return max(0.01, min(f, 0.25))
    
    def calculate_volatility_based_sizing(self, volatility: float, base_size: float = 0.1) -> float:
        """
        Reduce position size in high volatility, increase in low volatility
        """
        if volatility > 0.03:
            return base_size * 0.5
        elif volatility > 0.02:
            return base_size * 0.75
        else:
            return base_size
    
    def calculate_dynamic_stop_loss(self, entry_price: float, atr: float, risk_factor: float = 2.0) -> float:
        """Calculate dynamic stop-loss based on ATR"""
        stop_loss = entry_price - (atr * risk_factor)
        return max(stop_loss, entry_price * 0.95)  # Minimum 5% stop-loss
    
    def calculate_dynamic_take_profit(self, entry_price: float, atr: float, reward_factor: float = 3.0) -> float:
        """Calculate dynamic take-profit based on ATR"""
        take_profit = entry_price + (atr * reward_factor)
        return take_profit
    
    def calculate_pyramid_entry_size(self, signal_confidence: float, balance: float) -> float:
        """
        Pyramid entry: larger position on stronger signals
        """
        if signal_confidence > 0.90:
            return balance * 0.50  # 50% on very strong signal
        elif signal_confidence > 0.85:
            return balance * 0.35  # 35% on strong signal
        elif signal_confidence > 0.80:
            return balance * 0.25  # 25% on medium signal
        elif signal_confidence > 0.75:
            return balance * 0.15  # 15% on weak signal
        else:
            return balance * 0.05  # 5% on very weak signal
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                               highest_price: float, trailing_percent: float = 2.0) -> float:
        """
        Calculate trailing stop-loss
        """
        trailing_stop = highest_price * (1 - trailing_percent / 100)
        return max(trailing_stop, entry_price * 0.95)
    
    def generate_signal(self, predictions: np.ndarray, confidence: np.ndarray, 
                       volatility: float, atr: float, price: float) -> Tuple[str, float, Dict]:
        """
        Generate trading signal with risk management parameters
        """
        avg_confidence = np.mean(confidence)
        
        if predictions[0] == 1:
            signal = "BUY"
        else:
            signal = "SELL"
        
        # Calculate position size using multiple methods
        kelly_size = self.calculate_kelly_criterion(self.win_rate, self.avg_win, self.avg_loss)
        volatility_size = self.calculate_volatility_based_sizing(volatility)
        pyramid_size = self.calculate_pyramid_entry_size(avg_confidence, self.balance)
        
        # Average the sizing methods
        position_size = (kelly_size + volatility_size) / 2
        
        # Calculate risk management levels
        stop_loss = self.calculate_dynamic_stop_loss(price, atr)
        take_profit = self.calculate_dynamic_take_profit(price, atr)
        
        signal_data = {
            'signal': signal,
            'confidence': avg_confidence,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'kelly_criterion': kelly_size,
            'volatility_factor': volatility_size,
            'pyramid_size': pyramid_size,
            'atr': atr,
            'volatility': volatility
        }
        
        return signal, avg_confidence, signal_data
    
    def execute_trade(self, signal: str, price: float, position_size: float, 
                     stop_loss: float, take_profit: float, date: str) -> Dict:
        """Execute a trade with risk management"""
        
        if signal == "BUY":
            shares = int((self.balance * position_size) / price)
            if shares > 0:
                cost = shares * price
                self.balance -= cost
                
                trade = {
                    'date': date,
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'cost': cost,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size
                }
                
                self.trades.append(trade)
                return trade
        
        elif signal == "SELL" and len(self.trades) > 0:
            last_buy = None
            for trade in reversed(self.trades):
                if trade['type'] == 'BUY' and 'profit' not in trade:
                    last_buy = trade
                    break
            
            if last_buy:
                revenue = last_buy['shares'] * price
                profit = revenue - last_buy['cost']
                profit_percent = (profit / last_buy['cost']) * 100
                
                # Update win rate and average win/loss
                if profit > 0:
                    self.avg_win = (self.avg_win + profit) / 2
                else:
                    self.avg_loss = (self.avg_loss + abs(profit)) / 2
                
                self.balance += revenue
                
                trade = {
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': last_buy['shares'],
                    'revenue': revenue,
                    'profit': profit,
                    'profit_percent': profit_percent,
                    'position_size': position_size
                }
                
                last_buy['profit'] = profit
                last_buy['exit_price'] = price
                last_buy['exit_date'] = date
                
                self.trades.append(trade)
                return trade
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        total_trades = len([t for t in self.trades if t['type'] == 'SELL'])
        winning_trades = len([t for t in self.trades if t['type'] == 'SELL' and t.get('profit', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        total_profit = sum([t.get('profit', 0) for t in self.trades if t['type'] == 'SELL'])
        roi = (total_profit / self.initial_balance) * 100
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = self.initial_balance
        cumulative_balance = self.initial_balance
        
        for trade in self.trades:
            if trade['type'] == 'SELL':
                cumulative_balance += trade.get('profit', 0)
                drawdown = (peak - cumulative_balance) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
                peak = max(peak, cumulative_balance)
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for trade in self.trades:
            if trade['type'] == 'SELL':
                ret = trade.get('profit_percent', 0)
                returns.append(ret)
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'avg_trade_pnl': total_profit / total_trades if total_trades > 0 else 0
        }
        
        return metrics
