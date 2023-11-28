from src.Flight_Fare_estimator_Project.config.configuration import ConfigurationManager
from src.Flight_Fare_estimator_Project.components.data_transformation import DataTransformation
from src.Flight_Fare_estimator_Project import logger



STAGE_NAME="Data Transformation stage"


class DataTransformation_stage:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            dataframe=data_transformation.handle_missing_values()
            dataframe=data_transformation.outlier_treatment(dataframe)
            x,y=data_transformation.feature_classification(dataframe,'target_feature.csv')
            data_transformation.save_to_csv(x, 'x_output.csv')
            data_transformation.save_to_csv(y, 'y_output.csv')
            
        except Exception as e:
            raise e

if __name__=='__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
        obj=DataTransformation_stage()
        x_scaled_variable = obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
    except Exception as e:
        raise e
