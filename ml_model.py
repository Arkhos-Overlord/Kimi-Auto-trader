import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class MLModel:
    def __init__(self):
        self.model = None

    def train(self, data_path="data.csv"):
        print("Training machine learning model...")
        # Placeholder for data loading and preprocessing
        # In a real scenario, you would load historical market data
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: {data_path} not found. Please provide historical data for training.")
            # Create dummy data for demonstration
            data = {
                'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            }
            df = pd.DataFrame(data)
            df.to_csv(data_path, index=False)
            print(f"Created dummy data at {data_path} for demonstration.")

        X = df[['feature1', 'feature2']]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model trained with accuracy: {accuracy:.2f}")

    def predict(self, features):
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None
        # Placeholder for making predictions
        # 'features' would be a pandas DataFrame or similar structure
        prediction = self.model.predict(features)
        print(f"Prediction: {prediction[0]}")
        return prediction[0]
