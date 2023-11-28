from src.Flight_Fare_estimator_Project import logger
from src.Flight_Fare_estimator_Project.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.Flight_Fare_estimator_Project.pipeline.stage_2_data_preprocessing import Datapreprocessing_stage
from src.Flight_Fare_estimator_Project.pipeline.stage_3_data_transformation import DataTransformation_stage
from src.Flight_Fare_estimator_Project.pipeline.stage_4_data_modelling import DataModelling_stage






STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME="Data Preprocessing stage  "

try:
   logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
   obj=Datapreprocessing_stage()
   obj.main()
   logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
except Exception as e:
   logger.exception(e)
   raise e

STAGE_NAME="Data Transformation Stage  "

try:
   logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
   obj=DataTransformation_stage()
   obj.main()
   logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
except Exception as e:
   logger.exception(e)
   raise e


STAGE_NAME="Data Modelling Stage  "

try:
   logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
   obj=DataModelling_stage()
   obj.main()
   logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
except Exception as e:
   logger.exception(e)
   raise e



