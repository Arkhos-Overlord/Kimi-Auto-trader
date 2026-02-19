"""
Kimi-Auto-trader: Autonomous ML Trading Bot
Main execution loop with:
- Daily data updates (rolling window)
- Weekly model retraining
- Real-time accuracy monitoring
- Self-healing on performance degradation
- Continuous market signal generation
"""

import time
import logging
from datetime import datetime, timedelta
from config import Config
from exchange import Exchange
from strategy import TradingStrategy
from validator import AccuracyValidator
from daily_data_updater import DailyDataUpdater
from weekly_retrainer import WeeklyRetrainer

# Configure logging for autonomous operation
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutonomousTradingBot:
    """Main autonomous trading bot with continuous learning"""
    
    def __init__(self):
        """Initialize the bot with all components"""
        logger.info("=" * 70)
        logger.info("INITIALIZING KIMI-AUTO-TRADER AUTONOMOUS BOT")
        logger.info("=" * 70)
        
        self.config = Config()
        self.exchange = Exchange(self.config.API_KEY, self.config.API_SECRET)
        self.strategy = TradingStrategy(self.exchange)
        self.validator = AccuracyValidator(self.strategy.ml_model)
        
        # Data and retraining components
        self.data_updater = DailyDataUpdater()
        self.retrainer = WeeklyRetrainer()
        
        # Tracking variables
        self.cycle_count = 0
        self.last_daily_update = None
        self.last_weekly_retrain = None
        self.trading_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'successful_trades': 0,
            'failed_trades': 0
        }
        
        logger.info("✓ Bot initialized successfully")
        logger.info("=" * 70)
    
    def check_daily_update(self):
        """Check if daily data update should be performed"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.last_daily_update != today:
            logger.info("\n" + "=" * 70)
            logger.info("DAILY DATA UPDATE CHECK")
            logger.info("=" * 70)
            
            success = self.data_updater.update_daily()
            
            if success:
                self.last_daily_update = today
                stats = self.data_updater.get_update_statistics()
                if stats:
                    logger.info(f"Data window: {stats['date_range']}")
                    logger.info(f"Total rows: {stats['total_rows']}")
                    logger.info(f"Years of data: {stats['years_of_data']:.2f}")
                logger.info("✓ Daily update completed")
            else:
                logger.warning("✗ Daily update failed or skipped")
            
            logger.info("=" * 70 + "\n")
    
    def check_weekly_retrain(self):
        """Check if weekly model retraining should be performed"""
        should_retrain, reason = self.retrainer.should_retrain()
        
        if should_retrain:
            logger.info("\n" + "=" * 70)
            logger.info("WEEKLY MODEL RETRAINING")
            logger.info("=" * 70)
            logger.info(f"Reason: {reason}")
            
            result = self.retrainer.retrain_model()
            
            if result.get('status') == 'success':
                self.last_weekly_retrain = datetime.now().strftime('%Y-%m-%d')
                logger.info(f"Train Accuracy: {result.get('train_accuracy'):.4f}")
                logger.info(f"Test Accuracy: {result.get('test_accuracy'):.4f}")
                
                if result.get('improvement'):
                    logger.info(f"Improvement: {result.get('improvement'):+.4f}")
                
                logger.info("✓ Retraining completed successfully")
            else:
                logger.warning(f"✗ Retraining failed: {result.get('reason')}")
            
            logger.info("=" * 70 + "\n")
    
    def validate_performance(self):
        """Validate model performance and trigger self-healing if needed"""
        logger.info("[VALIDATION] Checking model performance...")
        
        # Get current accuracy
        validator_result = self.validator.validate_recent_performance()
        
        if self.validator.needs_retraining():
            logger.warning("[SELF-HEALING] Performance degradation detected!")
            logger.warning("[SELF-HEALING] Triggering immediate retraining...")
            
            result = self.retrainer.retrain_model()
            
            if result.get('status') == 'success':
                logger.info(f"[SELF-HEALING] ✓ Recovery successful - Accuracy: {result.get('test_accuracy'):.4f}")
                self.validator.reset_metrics()
            else:
                logger.error(f"[SELF-HEALING] ✗ Recovery failed: {result.get('reason')}")
    
    def execute_trading_cycle(self):
        """Execute a single trading cycle"""
        logger.info(f"\n--- TRADING CYCLE {self.cycle_count} ---")
        
        try:
            # 1. Daily data update check
            self.check_daily_update()
            
            # 2. Weekly retraining check
            self.check_weekly_retrain()
            
            # 3. Validate model performance
            self.validate_performance()
            
            # 4. Generate trading signals
            logger.info("[STRATEGY] Generating trading signals...")
            signals = self.strategy.execute_strategy()
            
            if signals:
                self.trading_stats['total_signals'] += len(signals)
                for signal in signals:
                    if signal.get('action') == 'BUY':
                        self.trading_stats['buy_signals'] += 1
                    elif signal.get('action') == 'SELL':
                        self.trading_stats['sell_signals'] += 1
                
                logger.info(f"[STRATEGY] Generated {len(signals)} signal(s)")
                for signal in signals:
                    logger.info(f"  - {signal.get('action')} @ {signal.get('price')} (confidence: {signal.get('confidence'):.2%})")
            else:
                logger.info("[STRATEGY] No signals generated (confidence too low)")
            
            # 5. Log cycle statistics
            logger.info(f"\n[STATS] Trading Statistics:")
            logger.info(f"  Total signals: {self.trading_stats['total_signals']}")
            logger.info(f"  BUY signals: {self.trading_stats['buy_signals']}")
            logger.info(f"  SELL signals: {self.trading_stats['sell_signals']}")
            
            logger.info(f"--- CYCLE {self.cycle_count} COMPLETE ---\n")
            
        except Exception as e:
            logger.error(f"[ERROR] Error in trading cycle: {str(e)}")
    
    def run_autonomous_loop(self, max_cycles=None, interval_seconds=3600):
        """
        Run the autonomous trading bot
        
        Args:
            max_cycles: Maximum number of cycles (None = infinite)
            interval_seconds: Seconds between cycles (default: 3600 = 1 hour)
        """
        logger.info("=" * 70)
        logger.info("STARTING AUTONOMOUS TRADING LOOP")
        logger.info(f"Interval: {interval_seconds} seconds ({interval_seconds/3600:.1f} hours)")
        if max_cycles:
            logger.info(f"Max cycles: {max_cycles}")
        else:
            logger.info("Max cycles: Unlimited (runs until interrupted)")
        logger.info("=" * 70)
        
        try:
            while True:
                self.cycle_count += 1
                
                # Execute trading cycle
                self.execute_trading_cycle()
                
                # Check if max cycles reached
                if max_cycles and self.cycle_count >= max_cycles:
                    logger.info(f"Max cycles ({max_cycles}) reached. Stopping bot.")
                    break
                
                # Wait for next cycle
                logger.info(f"Waiting {interval_seconds} seconds until next cycle...")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("\n" + "=" * 70)
            logger.info("BOT STOPPED BY USER")
            logger.info("=" * 70)
        except Exception as e:
            logger.error(f"Critical error in autonomous loop: {str(e)}")
            logger.error("=" * 70)
        finally:
            self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final statistics before shutdown"""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total cycles completed: {self.cycle_count}")
        logger.info(f"Total signals generated: {self.trading_stats['total_signals']}")
        logger.info(f"BUY signals: {self.trading_stats['buy_signals']}")
        logger.info(f"SELL signals: {self.trading_stats['sell_signals']}")
        logger.info(f"Last daily update: {self.last_daily_update}")
        logger.info(f"Last weekly retrain: {self.last_weekly_retrain}")
        logger.info("=" * 70)


def main():
    """Main entry point"""
    bot = AutonomousTradingBot()
    
    # Run autonomous loop
    # For demonstration: 5 cycles with 2 second intervals
    # For production: Remove max_cycles and set interval_seconds to 3600 (1 hour)
    bot.run_autonomous_loop(max_cycles=5, interval_seconds=2)


if __name__ == "__main__":
    main()
