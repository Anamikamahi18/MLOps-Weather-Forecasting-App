from src.components.data_ingestion import DataIngestion

from src.components.data_preprocessing import DataPreprocessing

from src.components.model_training import ModelTraining

from src.components.model_prediction import ModelPrediction

if __name__ == "__main__":

    data_ingestion = DataIngestion()
    data_ingestion.save_data()

    data_preprocessing = DataPreprocessing()
    data_preprocessing.save_processed_data()

    model_training = ModelTraining()
    model_training.save_model()

    model_prediction = ModelPrediction()
    model_prediction.predict_temperature()
