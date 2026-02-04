import logging
import pandas as pd
import os
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass
    
class ScalingType(Enum):
    MIN_MAX = "min_max"
    STANDARD = "standard"
    
class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def scale(self, df, columns):
        df[columns] = self.scaler.fit_transform(df[columns])
        self.is_fitted = True
        logging.info(f"Applied Min-Max Scaling on columns: {columns}")
        return df
    
    def get_scaler(self):
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet.")
        return self.scaler