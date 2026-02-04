import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class DataSplitStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame,
                   pd.DataFrame, pd.Series, pd.Series]:
        pass
    
    
class SplitType(str, Enum):
    SIMPLE = "simple"
    STRATIFIED = "stratified"
    
class SimpleDataSplitStrategy(DataSplitStrategy):
    def __init__(self, test_size):
        self.test_size = test_size
        
    def split_data(self, df, target_column):
        Y = df[target_column]
        X = df.drop(columns=[target_column])
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size = self.test_size, random_state=42
        )
        logging.info(f"Performed simple data split with test size = {self.test_size}")
        
        return X_train, X_test, Y_train, Y_test
