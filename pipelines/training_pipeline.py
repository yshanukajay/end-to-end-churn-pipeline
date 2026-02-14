import os
import sys
import joblib
import logging
import pandas as pd
from data_pipeline import data_pipeline
from typing import Dict, Any, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training import ModelTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_data_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def training_pipeline(
    data_path: str = "data/telco_data.csv",
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str = "artifacts/models/random_forest_model_cv_model.joblib",
):
    if not os.path.exists(get_data_path()['X_train_path']) and \
       not os.path.exists(get_data_path()['X_test_path']) and \
       not os.path.exists(get_data_path()['y_train_path']) and \
       not os.path.exists(get_data_path()['y_test_path']):
           
           data_pipeline()
    
    else:
        print("Data artifacts already exist. Skipping data pipeline and loading existing artifacts...")
        
training_pipeline()
           
        

