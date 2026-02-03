import logging
import pandas as pd
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bin_definitions):
        self.bin_definitions = bin_definitions
    
    def bin_features(self, df, col):
        def assign_bin(value):
            if value == 850:
                return 'Excellent'

            for bin_range, label in self.bin_definitions.items():
                if len(bin_range) == 2:
                    if bin_range[0] <= value <= bin_range[1]:
                        return label
                elif len(bin_range) == 1:
                    if value >= bin_range[0]:
                        return label
            
            return 'Invalid'

        df[f'{col}_binned'] = df[col].apply(assign_bin)
        logging.info(f"Binned feature '{col}' using custom bin definitions.")
        
        return df
