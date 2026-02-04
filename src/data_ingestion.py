import os
import pandas as pd
from abc import ABC, abstractmethod

class DataIngester(ABC):
    @abstractmethod
    def ingest_data(self, file_path_or_link: str) -> pd.DataFrame:
        pass
    
class DataIngestorCSV(DataIngester):
    def ingest_data(self, file_path_or_link: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path_or_link)
        except Exception as e:
            raise RuntimeError(f"Error ingesting CSV data: {e}")
        
class DataIngestorExcel(DataIngester):
    def ingest_data(self, file_path_or_link: str) -> pd.DataFrame:
        try:
            return pd.read_excel(file_path_or_link)
        except Exception as e:
            raise RuntimeError(f"Error ingesting Excel data: {e}")
        