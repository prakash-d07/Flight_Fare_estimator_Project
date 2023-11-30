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
from scipy.sparse import csr_matrix
from src.Flight_Fare_estimator_Project import logger


class DataModelling:
    def __init__(self,config):
        self.config=config
    

    def get_data_transformed_object(self):
        try:
            x = pd.read_csv(self.config.x_datapath)
            y = pd.read_csv(self.config.y_datapath)

            logger.info(f"Columns in x: {x.columns}")
            logger.info(f"{x.dtypes}")

            numerical_columns = ['Duration_minutes', 'Month_of_Month_of_journey', 'day_of_date_of_journey']
            categorical_columns = ['Airline', 'Source', 'Destination', 'Total_Stops']

            data_OHE = pd.concat([
                x[['Month_of_Month_of_journey', 'day_of_date_of_journey', 'Duration_minutes']],
                pd.get_dummies(x['Airline'].str.replace(' ', '_')),
                pd.get_dummies(x['Source'], prefix='source'),
                pd.get_dummies(x['Destination'], prefix='destination'),
                pd.get_dummies(x['Total_Stops'].str.replace(' ', '_'), prefix="stop")  # Replace spaces with underscores
            ], axis=1)


            data_OHE = data_OHE.astype(int)
            numerical_data = data_OHE[numerical_columns]
            scaler = StandardScaler()
            scaled_numerical_data = scaler.fit_transform(numerical_data)
            scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_columns)
            scaled_numerical_df.reset_index(drop=True, inplace=True)
            data_OHE.reset_index(drop=True, inplace=True)


            data_OHE.drop(columns=numerical_columns, inplace=True)

            scaled_data_OHE = pd.concat([scaled_numerical_df, data_OHE], axis=1)

            scaler_filename = self.config.preprocessor_file_path
            with open(scaler_filename, 'wb') as scaler_file:
                pickle.dump(scaler, scaler_file)

            logger.info(f"columns of preprocessed data   {scaled_data_OHE.columns}")
            logger.info(f"shape of preprocessed data     {scaled_data_OHE.shape}")


            return scaled_data_OHE, y
        except Exception as e:
            raise e

    def train_test_variables(self, x_scaled_variable, y):
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
    
