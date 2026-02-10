"""Gradient Boosting Model Trainer for Building Energy Prediction.

This module implements training pipelines using XGBoost and LightGBM for predicting
building energy consumption based on weather conditions, occupancy patterns, and HVAC data.
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for gradient boosting model."""
    model_type: str = "xgboost"  # xgboost, lightgbm, or sklearn
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_jobs: int = -1


class EnergyPredictionModel:
    """Gradient Boosting model for building energy consumption prediction."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.feature_importance = None
        self.metrics = {}
        
    def _build_xgboost_model(self) -> xgb.XGBRegressor:
        """Build XGBoost model with optimized hyperparameters."""
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            objective='reg:squarederror'
        )
    
    def _build_lightgbm_model(self) -> lgb.LGBMRegressor:
        """Build LightGBM model with optimized hyperparameters."""
        return lgb.LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=-1
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the gradient boosting model.
        
        Args:
            X: Feature matrix
            y: Target variable (energy consumption)
            
        Returns:
            Dictionary of training metrics
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build and train model
        if self.config.model_type == "xgboost":
            self.model = self._build_xgboost_model()
        elif self.config.model_type == "lightgbm":
            self.model = self._build_lightgbm_model()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Model trained with R² = {self.metrics['r2']:.4f}")
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self.model, f"{path}/model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(f"{path}/model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        logger.info(f"Model loaded from {path}")
