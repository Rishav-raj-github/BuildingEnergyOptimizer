"""Data preprocessing and feature engineering for building energy data.

Handles data cleaning, outlier detection, feature creation, and normalization.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class EnergyDataPreprocessor:
    """Preprocesses building energy consumption data."""
    
    def __init__(self):
        self.outlier_bounds = {}
        self.temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend']
        self.weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        self.building_features = ['occupancy_level', 'hvac_setpoint', 'equipment_status']
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'timestamp') -> pd.DataFrame:
        """Create temporal features from timestamp."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['hour'] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df
    
    def detect_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using z-score method."""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    def remove_outliers(self, df: pd.DataFrame, target_col: str = 'energy_consumption') -> pd.DataFrame:
        """Remove outliers from dataset."""
        outlier_mask = self.detect_outliers(df, target_col)
        return df[~outlier_mask].reset_index(drop=True)
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing values in time series data."""
        df = df.copy()
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        # Create temporal features
        df = self.create_temporal_features(df)
        # Handle missing values
        df = self.handle_missing_values(df)
        # Remove outliers
        df = self.remove_outliers(df)
        return df
