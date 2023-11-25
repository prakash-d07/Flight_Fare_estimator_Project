import warnings
import os
from pathlib import Path
import urllib.request as request
import zipfile
from src.Flight_Fare_estimator_Project import logger
from src.Flight_Fare_estimator_Project.utils.common import get_size
from src.Flight_Fare_estimator_Project.entity.config_entity import DatapreprocessConfig
import pandas as pd

class DataPreprocessing:
    def __init__(self, config:DatapreprocessConfig):
        self.config = config

    def read_data_frame(self):
        """
        Method: read_data_frame
        Description: This method is used to load the dataset into a pandas DataFrame.
        Parameters: None
        Return: DataFrame after loading the dataset
        Version: 1.0
        """
        try:
            logger.info("Loading the dataset into a pandas DataFrame")
            # Redirect warnings to the logging system
            warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
            self.df = pd.read_excel(self.config.datapath)
            logger.info("Raw data got loaded into the DataFrame")
            logger.info(f"DataFrame shape: {self.df.shape}")  # Log the shape
            return self.df

        except Exception as e:
            raise e

    def change_data_type(self, df, columns, change_type):
        """
        Method: change_data_type
        Description: This method is used to change the data type of specified columns in a DataFrame.
        Parameters:
            - df: DataFrame
            - columns: List of columns to change data type
            - change_type: Desired data type ('int', 'float', 'object', 'datetime')
        Return: DataFrame with updated data types
        Version: 1.0
        """
        try:
            logger.info("Changing data types of specified columns")
            for column in columns:
                if change_type == 'int':
                    df[column] = df[column].astype(int)
                elif change_type == 'float':
                    df[column] = df[column].astype(float)
                elif change_type == 'object':
                    df[column] = df[column].astype(object)
                elif change_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')

            logger.info("Data types changed successfully")
            return df

        except Exception as e:
            raise e

    def extract_month(self, df, cols):
        """
        Method: extracting month
        Description: This method is used to extract the month from the DataFrame
        Parameters:
            - df: DataFrame
            - cols: List of columns to extract the month
        Return: DataFrame with updated extracted months
        Version: 1.0
        """
        try:
            logger.info("Extracting months from the DataFrame")
            for col in cols:
                new_col_name = "Month_of_" + col
                if new_col_name not in df.columns:
                    df[new_col_name] = df["Date_of_Journey"].dt.month
            logger.info(f"Month extracted from the column {cols}")
            return df
        except Exception as e:
            raise e
    
    def extract_day(self, df, cols):
        """
        Method: extracting day
        Description: This method is used to extract the day from the DataFrame
        Parameters:
            - df: DataFrame
            - cols: List of columns to extract the month
        Return: DataFrame with updated extracted months
        Version: 1.0
        """
        try:
            logger.info("Extracting day from the DataFrame")
            for col in cols:
                new_col_name = "day_of_" + col
                if new_col_name not in df.columns:
                    df[new_col_name] = df["Date_of_Journey"].dt.day
            logger.info(f"day extracted from the column {cols}")
            return df
        except Exception as e:
            raise e


    def extract_year(self, df, cols):
        """
        Method: extracting year
        Description: This method is used to extract the year from the DataFrame
        Parameters:
            - df: DataFrame
            - cols: List of columns to extract the year
        Return: DataFrame with updated extracted years
        Version: 1.0
        """
        try:
            logger.info("Extracting years from the DataFrame")
            for col in cols:
                new_col_name = "Year_of_" + col
                if new_col_name not in df.columns:
                    df[new_col_name] = df["Date_of_Journey"].dt.year
            logger.info(f"Year extracted from the column {cols}")
            return df
        except Exception as e:
            raise e
    
    def extract_hour(self, df, cols):
        """
        Method: extracting hour
        Description: This method is used to extract the hour from the DataFrame
        Parameters:
            - df: DataFrame
            - cols: List of columns to extract the hour
        Return: DataFrame with updated extracted hours
        Version: 1.0
        """
        try:
            logger.info("Extracting hours from the DataFrame")
            for col in cols:
                new_col_name = "Hour_of_" + col
                if new_col_name not in df.columns:
                    df[new_col_name] = df[col].dt.hour
            logger.info(f"Hour extracted from the column {cols}")
            return df
        except Exception as e:
            raise e

    def extract_minute(self, df, cols):
        """
        Method: extracting minute
        Description: This method is used to extract the minute from the DataFrame
        Parameters:
            - df: DataFrame
            - cols: List of columns to extract the minute
        Return: DataFrame with updated extracted minutes
        Version: 1.0
        """
        try:
            logger.info("Extracting minutes from the DataFrame")
            for col in cols:
                new_col_name = "Minute_of_" + col
                if new_col_name not in df.columns:
                    df[new_col_name] = df[col].dt.minute
            logger.info(f"Minute extracted from the column {cols}")
            return df
        except Exception as e:
            raise e
        
    @staticmethod
    def duration_to_minutes(duration):
        parts = duration.strip().split()
        hours = 0
        minutes = 0
        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))
        return hours * 60 + minutes
    
    def extract_duration_minutes(self, df, duration_cols):
        try:
            logger.info("Converting duration from hours to minutes")
            for col in duration_cols:
                new_col_name = col + "_minutes"
                if new_col_name not in df.columns:
                    df[new_col_name] = df[col].apply(self.duration_to_minutes)
            logger.info(f"Duration converted to minutes for columns {duration_cols}")
            return df
        except Exception as e:
            raise e

    def drop_columns(self, df, cols):
        try:
            logger.info("Dropping the columns which are not required from the dataframe")
            
            # Check if columns exist before dropping
            existing_cols = set(df.columns)
            cols_to_drop = [col for col in cols if col in existing_cols]

            if cols_to_drop:
                df.drop(cols_to_drop, axis=1, inplace=True)
                logger.info(f"Dropped the {cols_to_drop} from the DataFrame The remaing Total columns are {df.columns}")
            else:
                logger.warning(f"Columns {cols} not found in the DataFrame. No columns dropped.")
            logger.info("Columns dropping method completed")
            return df
                
        except Exception as e:
            raise e
        
    def save_to_csv(self, df,file_path):
        """
        Method: save_to_csv
        Description: This method is used to save the DataFrame to a CSV file.
        Parameters:
            - df: DataFrame
            - file_path: Path to save the CSV file
        Return: None
        Version: 1.0
        """
        try:
            output_filepath = os.path.join(self.config.root_dir, file_path)
            if not os.path.exists(output_filepath):
                logger.info(f"Saving DataFrame to CSV file: {output_filepath}")
                df.to_csv(output_filepath, index=False)
                logger.info("DataFrame saved successfully.")
            else:
                logger.warning(f"CSV file already exists at {output_filepath}. Not saving.")
        except Exception as e:
            raise e