from src.Flight_Fare_estimator_Project.config.configuration import ConfigurationManager
from src.Flight_Fare_estimator_Project.components.data_modelling import DataModelling
from src.Flight_Fare_estimator_Project import logger



STAGE_NAME="Data Modelling stage"


class DataModelling_stage:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_modelling_config = config.get_modelling_config()
            data_modelling = DataModelling(config=data_modelling_config)
            x_scaled, y = data_modelling.get_data_transformed_object()
            x_train, x_test, y_train, y_test = data_modelling.train_test_variables(x_scaled, y)
            xgb_model = data_modelling.model_trainer(x_train, y_train, x_test, y_test)
            
        except Exception as e:
            raise e

if __name__=='__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started<<<<<")
        obj=DataModelling_stage()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed<<<<<")
    except Exception as e:
        raise e
