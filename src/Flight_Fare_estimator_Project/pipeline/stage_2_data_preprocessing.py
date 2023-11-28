from src.Flight_Fare_estimator_Project.config.configuration import ConfigurationManager
from src.Flight_Fare_estimator_Project.components.data_preprocessing import DataPreprocessing
from src.Flight_Fare_estimator_Project import logger



STAGE_NAME="Data Preprocessing stage"


class Datapreprocessing_stage:
    def __init__(self):
        pass

    def main(self):
        date_columns = ["Date_of_Journey", "Arrival_Time", "Dep_Time"]
        config = ConfigurationManager()
        data_preprocessing_config = config.get_datapreprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_frame = data_preprocessing.read_data_frame()
        data_frame = data_preprocessing.change_data_type(data_frame, date_columns, 'datetime')
        data_frame = data_preprocessing.extract_month(data_frame, ['Month_of_journey'])
        data_frame = data_preprocessing.extract_day(data_frame, ['date_of_journey'])
        data_frame = data_preprocessing.extract_hour(data_frame, ['Arrival_Time','Dep_Time'])
        data_frame = data_preprocessing.extract_minute(data_frame, ['Arrival_Time','Dep_Time'])
        data_frame = data_preprocessing.extract_duration_minutes(data_frame,['Duration'])
        data_frame = data_preprocessing.drop_columns(data_frame,['Duration','Date_of_Journey','Dep_Time','Arrival_Time','Route','Additional_Info','Hour_of_Arrival_Time','Hour_of_Dep_Time','Minute_of_Arrival_Time','Minute_of_Dep_Time'])
        data_preprocessing.save_to_csv(data_frame, 'output.csv')  


if __name__=='__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
        obj=Datapreprocessing_stage()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
    except Exception as e:
        raise e

