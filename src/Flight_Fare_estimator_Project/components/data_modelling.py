from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
import urllib.request as request
import zipfile
from src.Flight_Fare_estimator_Project import logger
from src.Flight_Fare_estimator_Project.utils.common import get_size
import pandas as pd
import numpy as np 
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import os
from src.Flight_Fare_estimator_Project import logger


class DataModelling:
    def __init__(self,config):
        self.config=config
    

    def get_data_transformed_object(self,file_path):
        """
        Method: To handle categorical vaiables and standardization
        Description: This method is used to standardize the data and handle categorical variables.
        Parameters: Outfile file path name
        Return: preprocessor_obj, Scaled Independent Features and output feature
        Version: 1.0
        """
        try:
            x=pd.read_csv(self.config.x_datapath)
            y=pd.read_csv(self.config.y_datapath)

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

            return preprocessor,scaled_x,y
        except Exception as e:
            raise e


    def train_test_variables(self, x_scaled_variable,y):
        """
        Method: Extracting Train and Test Variables
        Description: This method is used to extract Train and test variables from Dataframe.
        Parameters: dependent and Independent variable
        Return: dependent and Independent variables after Train test Split
        Version: 1.0
        """
        try:

            logger.info(f"Shape of data is X: {x_scaled_variable.shape},Y:{y.shape}")


            logger.info("Train Test Split of The Data Started")
            x_train, x_test, y_train, y_test = train_test_split(x_scaled_variable, y, test_size=0.2, random_state=42)
            logger.info(f"Train Test Split Completed. Shapes of training and test data:\n{x_train.shape}\n{x_test.shape}\n{y_train.shape}\n{y_test.shape}")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise e

    def model_trainer(self, x_train, y_train, x_test, y_test):
        """
        Method: Training the model from getting best models after research test
        Description: This method is used create best model.
        Parameters: train and test variables
        Return: model file
        Version: 1.0
        """

        try:
            logger.info("Training XGBoost Model Started")

            # Access XGBoost parameters from the configuration
            xgb_params = {
                'max_depth': self.config.max_depth,
                'max_features': self.config.max_features,
                'min_samples_leaf': self.config.min_samples_leaf,
                'min_samples_split': self.config.min_samples_split,
                'n_estimators': self.config.n_estimators
            }

            xgb_model = XGBRegressor(**xgb_params)
            xgb_model.fit(x_train, y_train)
            logger.info("Training XGBoost Model Completed")

            xgb_model_filepath = os.path.join(self.config.root_dir, 'xgb_model.pkl')
            with open(xgb_model_filepath, 'wb') as file:
                pickle.dump(xgb_model, file)

            logger.info(f"XGBoost model saved to {xgb_model_filepath}")
            return xgb_model
        except Exception as e:
            raise e