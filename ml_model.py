import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

class MLModel:
    def __init__(self, data_path="data.csv"):
        self.model = None
        self.data_path = data_path

    def train(self):
        print(f"Training machine learning model with advanced features from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} not found. Please ensure data is fetched and processed.")
            return

        df = pd.read_csv(self.data_path)
        
        # Select advanced features
        feature_cols = ['return', 'volatility', 'rsi', 'ema9', 'ema21', 'ema_signal']
        X = df[feature_cols]
        y = df['target']

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Using RandomForest with optimized settings
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model trained. Accuracy on test set: {accuracy:.2f}")
        return accuracy

    def predict(self, features):
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        
        # Ensure features are in the correct format (DataFrame with correct columns)
        prediction = self.model.predict(features)
        return prediction[0]
