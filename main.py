import time
import logging
from config import Config
from exchange import Exchange
from strategy import TradingStrategy
from validator import AccuracyValidator

# Configure logging for autonomous operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

def main():
    config = Config()
    logging.info("Initializing Autonomous Kimi-Auto-trader...")

    exchange = Exchange(config.API_KEY, config.API_SECRET)
    strategy = TradingStrategy(exchange)
    validator = AccuracyValidator(strategy.ml_model)

    # Autonomous Execution Loop
    # In a real scenario, this would run 24/7 or during market hours
    iteration = 0
    max_iterations = 5 # Set to a small number for sandbox demonstration
    
    try:
        while iteration < max_iterations:
            iteration += 1
            logging.info(f"--- Autonomous Cycle {iteration} Started ---")
            
            # 1. Validate previous predictions
            validator.validate_recent_performance()
            
            # 2. Check if self-healing (re-training) is needed
            if validator.needs_retraining():
                logging.warning("Performance dip detected. Triggering self-healing (re-training)...")
                strategy.ml_model.train()
                validator.reset_metrics()

            # 3. Execute Trading Strategy
            strategy.execute_strategy()
            
            # 4. Wait for next market interval (e.g., 1 minute, 1 hour, or 1 day)
            # For demo purposes, we use a short sleep
            logging.info("Cycle complete. Waiting for next interval...")
            time.sleep(2) 
            
    except KeyboardInterrupt:
        logging.info("Autonomous operation stopped by user.")
    except Exception as e:
        logging.error(f"Critical error in autonomous loop: {e}")

    logging.info("Kimi-Auto-trader demonstration finished.")

if __name__ == "__main__":
    main()
