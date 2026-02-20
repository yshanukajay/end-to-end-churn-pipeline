import json
import logging
import os
import sys
import joblib
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger=logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.binning_config = get_binning_config()
        self.encoding_config = get_encoding_config()
        self.encoders = {}

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.info(f"Model file not found at {self.model_path}. Please ensure the model is trained and saved correctly.")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        return joblib.load(self.model_path)

    def load_encoders(self, encoders_dir):
        for file in os.listdir(encoders_dir):
            feature_name = file.split('_encoder.json')[0]
            with open(os.path.join(encoders_dir, file), 'r') as f:
                self.encoders[feature_name] = json.load(f)

    def preprocess_input(self, input_data):
        data = pd.DataFrame([input_data])

        for col, encoder in self.encoders.items():
            data[col] = data[col].map(encoder)

        binning = CustomBinningStrategy(self.binning_config['credit_score_bins'])
        data = binning.bin_feature(data, 'CreditScore') 

        ordinal_encoding = OrdinalEncodingStrategy(self.encoding_config['ordinal_mappings'])
        data = ordinal_encoding.encode(data)

        data = data.drop(columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname', 'CreditScore'])
        print(data)   
        return data
    
    def predict(self, input_data):        
        preprocessed_data = self.preprocess_input(input_data)
        Y_pred = self.model.predict(preprocessed_data)
        Y_pred_proba = self.model.predict_proba(preprocessed_data)[:, 1]

        status = 'Churn' if Y_pred[0] == 1 else 'No Churn'
        logger.info(f"Predicted status: {status} with probability of churn: {Y_pred_proba[0]:.4f}")

        return {
            "prediction": int(Y_pred[0]),
            "Confidence": float(Y_pred_proba[0])
        }
 

      