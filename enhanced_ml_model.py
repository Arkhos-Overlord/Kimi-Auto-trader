"""
Enhanced ML Model with Ensemble Learning and Optimization
Implements XGBoost, LightGBM, Voting Classifier, and Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, Dict, Any

class EnhancedMLModel:
    """Enhanced ML model with ensemble learning and optimization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
    
    def create_xgboost_model(self) -> xgb.XGBClassifier:
        """Create optimized XGBoost model"""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    def create_lightgbm_model(self) -> lgb.LGBMClassifier:
        """Create optimized LightGBM model"""
        return lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=0.1,
            lambda_l2=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    def create_random_forest_model(self) -> RandomForestClassifier:
        """Create optimized Random Forest model"""
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    
    def create_gradient_boosting_model(self) -> GradientBoostingClassifier:
        """Create optimized Gradient Boosting model"""
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    
    def create_voting_ensemble(self) -> VotingClassifier:
        """Create Voting Classifier ensemble"""
        xgb_model = self.create_xgboost_model()
        lgb_model = self.create_lightgbm_model()
        rf_model = self.create_random_forest_model()
        
        voting = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        return voting
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Train all ensemble models"""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {}
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = self.create_xgboost_model()
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgb'] = xgb_model
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            xgb_score = xgb_model.score(X_val_scaled, y_val)
            results['xgb'] = xgb_score
            print(f"  XGBoost Validation Accuracy: {xgb_score:.4f}")
        
        # Train LightGBM
        print("Training LightGBM...")
        lgb_model = self.create_lightgbm_model()
        lgb_model.fit(X_train_scaled, y_train)
        self.models['lgb'] = lgb_model
        
        if X_val is not None:
            lgb_score = lgb_model.score(X_val_scaled, y_val)
            results['lgb'] = lgb_score
            print(f"  LightGBM Validation Accuracy: {lgb_score:.4f}")
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = self.create_random_forest_model()
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf'] = rf_model
        
        if X_val is not None:
            rf_score = rf_model.score(X_val_scaled, y_val)
            results['rf'] = rf_score
            print(f"  Random Forest Validation Accuracy: {rf_score:.4f}")
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = self.create_gradient_boosting_model()
        gb_model.fit(X_train_scaled, y_train)
        self.models['gb'] = gb_model
        
        if X_val is not None:
            gb_score = gb_model.score(X_val_scaled, y_val)
            results['gb'] = gb_score
            print(f"  Gradient Boosting Validation Accuracy: {gb_score:.4f}")
        
        # Train Voting Ensemble
        print("Training Voting Ensemble...")
        voting_model = self.create_voting_ensemble()
        voting_model.fit(X_train_scaled, y_train)
        self.models['voting'] = voting_model
        self.best_model = voting_model
        
        if X_val is not None:
            voting_score = voting_model.score(X_val_scaled, y_val)
            results['voting'] = voting_score
            print(f"  Voting Ensemble Validation Accuracy: {voting_score:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using best model"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from all models"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
            elif hasattr(model, 'estimators_'):
                # For voting classifier, get average importance
                importances = []
                for est_name, est in model.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        importances.append(est.feature_importances_)
                if importances:
                    importance_dict[name] = np.mean(importances, axis=0)
        
        return importance_dict
    
    def get_top_features(self, feature_names: list, top_n: int = 20) -> Dict[str, list]:
        """Get top N important features"""
        importance = self.get_feature_importance()
        top_features = {}
        
        for model_name, importances in importance.items():
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features[model_name] = [
                (feature_names[i], importances[i]) 
                for i in top_indices
            ]
        
        return top_features
