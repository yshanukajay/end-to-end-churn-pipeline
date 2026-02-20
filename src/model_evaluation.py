import os
import logging
import warnings
from typing import Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}

    def evaluate(self, X_test, Y_test):
        logger.info(f"Evaluating model: {self.model_name}")
        Y_pred = self.model.predict(X_test)

        cm = confusion_matrix(Y_test, Y_pred)
        acc_score = accuracy_score(Y_test, Y_pred)
        prec_score = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        
        self.evaluation_results = {
            "confusion_matrix": cm,
            "accuracy": acc_score, 
            "precision": prec_score,
            "recall": recall,
            "f1_score": f1
        }
        return self.evaluation_results

     