import os
from pyexpat import model
import sys
import joblib
import logging
import pandas as pd
from data_pipeline import data_pipeline
from typing import Dict, Any, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import RandomForestModelBuilder, XGBoostModelBuilder

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_data_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def training_pipeline(
    data_path: str = "data/telco_data.csv",
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str = "artifacts/models/churn_analysis_model.joblib",
):
    if not os.path.exists(get_data_path()['X_train_path']) and \
       not os.path.exists(get_data_path()['X_test_path']) and \
       not os.path.exists(get_data_path()['y_train_path']) and \
       not os.path.exists(get_data_path()['y_test_path']):
           
           data_pipeline()
    
    else:
        print("Data artifacts already exist. Skipping data pipeline and loading existing artifacts...")

    logging.info("Starting training pipeline...")

    # Load data
    X_train = pd.read_csv(get_data_path()['X_train_path'])
    y_train = pd.read_csv(get_data_path()['Y_train_path'])
    X_test = pd.read_csv(get_data_path()['X_test_path'])
    y_test = pd.read_csv(get_data_path()['Y_test_path'])

    # Build model
    model_builder = XGBoostModelBuilder(**get_model_config()['model_params'])
    model = model_builder.build_model()

    # Train model
    trainer = ModelTrainer()
    model, train_score = trainer.train(model, X_train, y_train.squeeze())
    print(model_params)
    logger.info(f"Training completed with training score: {train_score:.4f}")

    #save model
    trainer.save_model(model, model_path)
    logger.info(f"Model saved to {model_path}")

    #evaluate model
    evaluater = ModelEvaluator(model, "XGBoost")
    results = evaluater.evaluate(X_test, y_test)
    logger.info(f"Evaluation results: {results}")

        
if __name__ == "__main__":
    training_pipeline(model_params=get_model_config()['model_params'])
           
        

