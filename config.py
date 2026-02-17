import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("API_KEY")
    API_SECRET = os.getenv("API_SECRET")
    # Add other configuration parameters here

    def __init__(self):
        if not self.API_KEY or not self.API_SECRET:
            print("WARNING: API_KEY or API_SECRET not set in environment variables.")

