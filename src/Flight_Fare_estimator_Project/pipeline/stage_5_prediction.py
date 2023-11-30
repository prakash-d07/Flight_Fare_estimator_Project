import pickle 
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.Flight_Fare_estimator_Project import logger
from src.Flight_Fare_estimator_Project.utils.common import load_object

class PredictionPipeline:
    def __init__(self):
        model_path = Path('artifacts/data_modelling/xgb_model.pkl')
        preprocessor_path = Path('artifacts/data_modelling/scaler.pkl')

        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

        with open(preprocessor_path, 'rb') as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)

    def predict(self, features):
        try:
            logger.info("Before Loading")
            logger.info("After Loading")
            logger.info("Performing Preprocessing")
            data_scaled = self.preprocessor.transform(features)
            logger.info("Preprocessing completed. Next Prediction stage")
            preds = self.model.predict(data_scaled)
            logger.info("Returning the features after prediction")
            return preds

        except Exception as e:
            raise e



class CustomData:
    def __init__(  self,
        Airline: str,
        Source: str,
        Destination : str,
        Total_Stops: str,
        Journey_month: int,
        Journey_day: int,
        Total_Duration: int):

        self.Airline = Airline

        self.Source = Source

        self.Destination = Destination

        self.Total_Stops = Total_Stops

        self.Journey_month = Journey_month

        self.Journey_day = Journey_day

        self.Total_Duration = Total_Duration

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Airline": [self.Airline],
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Total_Stops": [self.Total_Stops],
                "Month_of_Month_of_journey": [self.Journey_month],
                "day_of_date_of_journey": [self.Journey_day],
                "Duration_minutes": [self.Total_Duration],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise e      
    