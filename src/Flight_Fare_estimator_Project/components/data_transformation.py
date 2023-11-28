import warnings
import os
import urllib.request as request
import zipfile
from src.Flight_Fare_estimator_Project import logger
from src.Flight_Fare_estimator_Project.utils.common import get_size
import pandas as pd
import numpy as np 
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def handle_missing_values(self):
        """
        Method: handle_missing_values
        Description: This method is used to handle missing values in Dataframe.
        Parameters: None
        Return: DataFrame after missing the dataset
        Version: 1.0
        """
        try:
            logger.info("Loading the Preprocessed DataFrame")
            # Redirect warnings to the logging system
            warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
            self.df = pd.read_csv(self.config.datapath)
            logger.info(f"Our Data frame consists \n {self.df.isnull().sum()} so Here we can drop the na values ")
            self.df.dropna(inplace=True)
            logger.info("Dropped the nan values ")
            logger.info(f"Null values are handling na values is: {self.df.isnull().sum()}")  # Log the shape
            return self.df

        except Exception as e:
            raise e
        
    def outlier_treatment(self,df):
        """
        Method: Handling Outliers
        Description: This method is used to handle Outliers values in Dataframe.
        Parameters: dataframe
        Return: DataFrame after Handling outliers the dataset
        Version: 1.0
        """
        try:
            logger.info("Calaculating IQR to eliminate outliers from Price variable")
            Q1 = df.Price.quantile(0.25)
            Q3 = df.Price.quantile(0.75)
            IQR = Q3 - Q1
            logger.info(f"IQR for the Outlier treatment Vaiable Price is {IQR}")
            lower_bridge_ = Q1 - (IQR * 3)
            upper_bridge_ = Q3 + (IQR * 3)
            df.loc[df['Price'] > upper_bridge_, 'Price']
            logger.info("Handled the outliers of price variable")
            return df
        except Exception as e:
            raise e
        

    def feature_classification(self, df,file_path):
        """
        Method: classifying Dependent and Independent variables
        Description: This method is used to classify Dependent and Independent variables in Dataframe.
        Parameters: dataframe
        Return: dependent avraibles and Independent variables
        Version: 1.0
        """
        try:
            logger.info("Classification of Independent and Dependent variable")
            x = df.drop(columns=['Price'], axis=1)
            y = df['Price']
            logger.info(f"Dependent variables are {x.columns}, and Independent variables are {y.name}")
          

            output_filepath = os.path.join(self.config.root_dir, file_path)

            if not os.path.exists(output_filepath):
                pd.DataFrame(y, columns=['Price']).to_csv(output_filepath, index=False)
                logger.info(f"Saving DataFrame to CSV file: {output_filepath}")
            else:
                logger.warning(f"CSV file already exists at {output_filepath}. Not saving.")

            return x, y
        except Exception as e:
            raise e



    def get_data_transformed_object(self, x, file_path):
        try:
            numerical_columns = ['Duration_minutes']
            categorical_columns = ['Airline', 'Source', 'Destination', 'Total_Stops']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            scaled_x = preprocessor.fit_transform(x)
            pickle_filepath = os.path.join(self.config.root_dir, file_path)
            with open(pickle_filepath, 'wb') as file:
                pickle.dump(scaled_x, file)

            return preprocessor,scaled_x
        except Exception as e:
            raise e