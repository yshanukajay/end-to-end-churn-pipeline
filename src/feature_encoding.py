import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import List, Dict
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  
) 

class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class variableType(Enum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    
class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_features):
        self.nominal_features = nominal_features
        self.encoder_dictionary = {}
        os.makedirs("artifacts/encoders", exist_ok=True)
        
    def encode(self, df):
        for col in self.nominal_features:
            unique_values = df[col].unique()
            encoder_dict = {val: idx for idx, val in enumerate(unique_values)}
            self.encoder_dictionary[col] = encoder_dict
            
            encoder_path = os.path.join('artifacts/encoders', f'{col}_encoder.json')
            with open(encoder_path, 'w') as file:
                json.dump(encoder_dict, file)
                
            df[col] = df[col].map(encoder_dict)
        return df
    
    
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings
            
    def encode(self, df):
        for col, mapping in self.ordinal_mappings.items():
            df[col] = df[col].map(mapping)
            logging.info(f"Encoded ordinal feature '{col}' with mapping: {mapping}") 
        return df
        