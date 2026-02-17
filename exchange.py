import requests
import os
import logging
from config import Config

class Exchange:
    def __init__(self, api_key, api_secret=None):
        """
        Initializes the IndMoney (INDstocks) exchange integration.
        api_key: The Access Token generated from the IndMoney dashboard.
        """
        self.access_token = api_key
        self.base_url = "https://api.indstocks.com"
        self.headers = {
            'Authorization': self.access_token,
            'Content-Type': 'application/json'
        }

    def get_balance(self):
        """Fetches the current account balance from IndMoney."""
        logging.info("Fetching account balance from IndMoney...")
        try:
            # Placeholder endpoint based on general portfolio APIs
            response = requests.get(f"{self.base_url}/portfolio/holdings", headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                # Simplified return for demonstration
                return data.get('data', {})
            else:
                logging.error(f"Failed to fetch balance: {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error fetching balance: {e}")
            return None

    def place_order(self, security_id, side, qty, order_type='MARKET', limit_price=None, product='CNC'):
        """
        Places an order on IndMoney.
        security_id: The ID of the instrument (e.g., '2885' for Reliance)
        side: 'BUY' or 'SELL'
        qty: Quantity to trade
        order_type: 'MARKET', 'LIMIT', etc.
        limit_price: Required for LIMIT orders
        product: 'CNC' (delivery), 'INTRADAY', or 'MARGIN'
        """
        logging.info(f"Placing {side} {order_type} order for {qty} shares of ID {security_id} via IndMoney...")
        
        order_data = {
            'txn_type': side,
            'exchange': 'NSE',
            'segment': 'EQUITY',
            'security_id': str(security_id),
            'qty': int(qty),
            'order_type': order_type,
            'validity': 'DAY',
            'product': product,
            'is_amo': False,
            'algo_id': '99999' # Default for regular orders
        }

        if order_type == 'LIMIT' and limit_price:
            order_data['limit_price'] = float(limit_price)

        try:
            response = requests.post(f"{self.base_url}/order", headers=self.headers, json=order_data)
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Order placed successfully! Order ID: {result['data']['order_id']}")
                return result
            else:
                logging.error(f"Order failed: {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return None

    def get_order_status(self, order_id):
        """Checks the status of a specific order."""
        try:
            response = requests.get(f"{self.base_url}/order-book", headers=self.headers)
            if response.status_code == 200:
                orders = response.json().get('data', [])
                for order in orders:
                    if order.get('id') == order_id:
                        return order.get('status')
            return "NOT_FOUND"
        except Exception as e:
            logging.error(f"Error checking order status: {e}")
            return "ERROR"
