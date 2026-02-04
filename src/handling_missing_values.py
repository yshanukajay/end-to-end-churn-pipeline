import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from prompt_toolkit import prompt
from pydantic import BaseModel
from abc import ABC, abstractmethod

logging.basicConfig(
    level = logging.INFO, 
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

load_dotenv()

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle_missing_values(self, df:pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns = []):
        self.critical_columns = critical_columns
        logging.info(f"Initialized DropMissingValues with critical columns: {self.critical_columns}")
    
    def handle_missing_values(self, df:pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.dropna(subset = self.critical_columns)
        logging.info(f"Dropped rows with missing values in columns: {self.critical_columns}")
        
        
        
class Gender(str, Enum):
    MALE = 'Male'
    FEMALE = 'Female'
    
    
class GenderPrediction(BaseModel):
    firstname: str
    lastname: str
    predicted_gender: Gender
    
    def predict_gender(
        firstname: str,
        lastname: str
    ):
        prompt = f"""
                    What is the most probable gender(Male/Female) for a person
                    with the first name '{firstname}' and last name '{lastname}'?  
                    
                    Your answer should consist of only one word: 'Male' or 'Female'.
                """   
                
                
        response = groq.Groq().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        
        pred_gender = response.choices[0].message.content.strip()
        prediction = GenderPrediction(firstname=firstname, lastname=lastname, predicted_gender=pred_gender)
        
        logging.info(f"Predicted gender for {firstname} {lastname}: {pred_gender}")
        
        return prediction.predicted_gender
    
    def impute(self, df):
        missing_gender_index = df.Gender.isnull()
        
        for idx in df[missing_gender_index].index:
            first_name = df.loc[idx, 'FirstName']
            last_name = df.loc[idx, 'LastName']
            gender = self.predict_gender(first_name, last_name)
            
            if gender:
                df.loc[idx, 'Gender'] = gender
                print(f"Imputed gender for {first_name} {last_name} as {gender}")
            else:
                print(f"Could not predict gender for {first_name} {last_name}")
        
        return df
    
class fillingMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(
        self,
        method = 'mean',
        fill_value = None,
        relevant_column = None,
        is_custom_imputer = False,
        custom_imputer = None
    ):
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer
        
        logging.info(
            f"Initialized fillingMissingValuesStrategy with method: {self.method}, fill_value: {self.fill_value}, relevant_column: {self.relevant_column}, is_custom_imputer: {self.is_custom_imputer}"
        )
        
        def handle_missing_values(self):
            if self.is_custom_imputer:
                df = self.custom_imputer.impute(df)
                logging.info("Applied custom imputer for missing values.")
                
            df[self.relevant_column] = df[self.relevant_column].fillna(df[self.relevant_column].mean())
            logging.info(f"Filled missing values in column {self.relevant_column} using method: {self.method}")
            return df
        
     
            
    