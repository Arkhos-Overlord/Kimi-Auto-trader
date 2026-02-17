from config import Config
from exchange import Exchange
from strategy import TradingStrategy

def main():
    config = Config()
    if not config.API_KEY or not config.API_SECRET:
        print("Please set API_KEY and API_SECRET in your .env file.")
        return

    exchange = Exchange(config.API_KEY, config.API_SECRET)
    strategy = TradingStrategy(exchange)

    print("Kimi-Auto-trader started.")
    strategy.execute_strategy()
    print("Kimi-Auto-trader finished.")

if __name__ == "__main__":
    main()
