import os
import sys
import pandas as pd
import numpy as np
from typing import Dict
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handling_missing_values import DropMissingValuesStrategy, fillingMissingValuesStrategy, GenderImputer
from outlier_detection import IQROutlierDetection, OutlierDetector
from feature_binning import CustomBinningStrategy
from feature_encoding import NominalEncodingStrategy, OrdinalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleDataSplitStrategy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import (get_data_path,
                    get_columns,
                    get_missing_value_config,
                    get_outlier_config,
                    get_binning_config,
                    get_encoding_config,
                    get_scaling_config,
                    get_split_config
) 

def data_pipeline(
    data_path: str = "data/telco_data.csv",
    target_column: str = "Exited",
    test_size: float = 0.2,
    force_build: bool = False
) -> Dict[str, np.ndarray]:
    
    data_paths_config = get_data_path()
    columns_config = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_split_config()
    
    
    print("Starting Data Ingestion...")
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths_config['artifacts_dir']) 
    X_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    X_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'y_test.csv')
    
    if os.path.exists(X_train_path) and \
       os.path.exists(X_test_path) and \
       os.path.exists(y_train_path) and \
       os.path.exists(y_test_path):
           
           X_train = pd.read_csv(X_train_path)
           X_test = pd.read_csv(X_test_path)
           y_train = pd.read_csv(y_train_path)
           y_test = pd.read_csv(y_test_path)  
           
           
    if not os.path.exists('temp_imputed_data.csv'):       
        ingester = DataIngestorCSV()     
        df = ingester.ingest_data(data_path)
        print("Data Ingestion Completed.")
        print(f"Data Shape: {df.shape}")
        
        
        print('Handling Missing Values...')
        drop_handler = DropMissingValuesStrategy(critical_columns=columns_config['critical_columns'])
        
        age_handler = fillingMissingValuesStrategy(
            method='mean',
            relevant_column='Age'
        )
        
        gender_handler = fillingMissingValuesStrategy(
            relevant_column='Gender',
            is_custom_imputer=True,
            custom_imputer=GenderImputer()
        )
        
        df = drop_handler.handle_missing_values(df)
        df = age_handler.handle_missing_values(df)
        df = gender_handler.handle_missing_values(df)
        df.to_csv('temp_imputed_data.csv')
    
    df = pd.read_csv('temp_imputed_data.csv')
    print("Missing Value Handling Completed.")
    
    print('Detecting and Handling Outliers...')
    
    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns_config['outlier_columns'])
    print(f'After Outlier Handling, Data Shape: {df.shape}')
    print("Outlier Detection and Handling Completed.")
    
data_pipeline()
        