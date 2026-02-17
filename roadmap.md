# Kimi-Auto-trader: Autonomous Roadmap (IndMoney Edition)

This roadmap focuses on the autonomous evolution of the Kimi-Auto-trader using the IndMoney (INDstocks) API.

## Current State: Autonomous V1
*   **IndMoney Integration**: `exchange.py` is configured to communicate with the INDstocks API for order placement and balance checks.
*   **Autonomous Loop**: `main.py` implements a continuous execution cycle with error handling and logging.
*   **Accuracy Validation**: `validator.py` monitors model performance and triggers self-healing (re-training) if accuracy dips below 70%.
*   **ML Engine**: Uses a Random Forest Classifier trained on historical Nifty 50 data with technical indicators (RSI, EMA).

## Phase 1: Real-time Market Data & Advanced Execution
*   **WebSocket Integration**: Implement real-time price streaming via IndMoney WebSockets for millisecond-level decision making.
*   **Dynamic Security ID Discovery**: Automate the mapping of stock symbols to IndMoney Security IDs using the Instruments API.
*   **Smart Order Types**: Implement GTT (Good Till Triggered) and Trailing Stop Loss orders for better risk management.

## Phase 2: Performance & Self-Healing Optimization
*   **Multi-Model Ensemble**: Use multiple ML models (e.g., XGBoost + LSTM) and aggregate their signals for higher reliability.
*   **Automated Data Refresh**: Periodically fetch the latest market data from NSE/BSE to keep the training set fresh.
*   **Advanced Validator**: Implement a "Shadow Mode" where the bot simulates trades for a period before going live with a new model version.

## Phase 3: Risk Management & Dashboard
*   **Position Sizing Engine**: Automatically calculate trade size based on account balance and risk-per-trade settings.
*   **Web Dashboard**: A simple UI to monitor the bot's status, recent trades, and performance metrics in real-time.
*   **Telegram/WhatsApp Alerts**: Instant notifications for trade executions and critical system events.

---
**Security Warning**: Always keep your IndMoney Access Token secure in your `.env` file. Never share it or commit it to GitHub.
