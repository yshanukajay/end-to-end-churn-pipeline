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
    
    
    print("Step 00 : Starting Data Ingestion...")
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
        print("\nData Ingestion Completed.")
        print(f"Data Shape: {df.shape}")
        
        
        print('\nStep 01 : Handling Missing Values...')
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
        df.to_csv('temp_imputed_data.csv', index=False)
    
    df = pd.read_csv('temp_imputed_data.csv')
    print("\nMissing Value Handling Completed.")
    
    
    print('\nStep 02 : Detecting and Handling Outliers...')
    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns_config['outlier_columns'])
    print(f'After Outlier Handling, Data Shape: {df.shape}')
    print("\nOutlier Detection and Handling Completed.")
    
    
    print('\nStep 03 : Binning Features...')
    binning = CustomBinningStrategy(binning_config['credit_score_bins'])
    df = binning.bin_feature(df, 'CreditScore')
    print("\nFeature Binning Completed.")
    print(f"Data after Binning:\n {df.head()}")
    
    
    print('\nStep 04 : Feature Encoding...')
    nominal_encoding = NominalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_encoding = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])
    df = nominal_encoding.encode(df)
    df = ordinal_encoding.encode(df)
    print("\nFeature Encoding Completed.")
    print(f"Data after Encoding:\n {df.head()}")
    
    
    print('\nStep 05 : Feature Scaling...')
    min_max_scaler = MinMaxScalingStrategy()   
    df = min_max_scaler.scale(df, scaling_config['columns_to_scale'])
    print("\nFeature Scaling Completed.")
    print(f"Data after Scaling:\n {df.head()}")
    
    
    print('\nStep 06 : Post processing...')
    df = df.drop(columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname', 'CreditScore'])
    print(f"Data after Post processing:\n {df.head()}")
    
    
    print('Step 07 : Data Splitting...')
    splitting = SimpleDataSplitStrategy(test_size=splitting_config['test_size'])
    X_train, X_test, y_train, y_test = splitting.split_data(df, target_column)
    print("\nData Splitting Completed.")
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    
data_pipeline()
        