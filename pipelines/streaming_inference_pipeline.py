import os
import sys
import json
import logging  
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_inference_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting streaming inference pipeline...")

inference = ModelInference(model_path="artifacts/models/churn_analysis_model.joblib")
   
def streaming_inference(inference, input_data):
    
    inference.load_encoders('artifacts/encoders')
    #preprocessed_data = inference.preprocess_input(input_data)
    #print(preprocessed_data)
    result = inference.predict(input_data)
    return result

if __name__ == "__main__":
    data = {
    "RowNumber": 6,
    "CustomerId": 15574012,
    "Firstname": "Jack",
    "Lastname": "Smith",
    "CreditScore": 645,
    "Geography": "Spain",
    "Gender": "Male",
    "Age": 44,
    "Tenure": 8,
    "Balance": 113755.78,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 149756.71,
    }

    pred = streaming_inference(inference, data)
    print(pred)