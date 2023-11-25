from src.Flight_Fare_estimator_Project import logger
from src.Flight_Fare_estimator_Project.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.Flight_Fare_estimator_Project.pipeline.stage_2_data_preprocessing import Datapreprocessing_stage_1




STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME="Data Preprocessing stage_1  "

try:
   logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
   obj=Datapreprocessing_stage_1()
   obj.main()
   logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
except Exception as e:
   logger.exception(e)
   raise e
