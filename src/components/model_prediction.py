import pandas as pd
import joblib


class ModelPrediction:

    def __init__(self):

        self.model_path = "models/weather_model.pkl"

        self.processed_data_path = "data/processed/" "weather_processed.csv"

    def load_model(self):

        model = joblib.load(self.model_path)

        return model

    def load_latest_data(self):

        dataframe = pd.read_csv(self.processed_data_path)

        latest_row = dataframe.iloc[-1:]

        return latest_row

    def predict_temperature(self):

        model = self.load_model()

        latest_data = self.load_latest_data()

        features = [
            "rain",
            "relative_humidity_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "weather_code",
            "month",
            "day",
            "hour",
            "day_of_week",
            "previous_temperature",
            "temperature_rolling_mean",
        ]

        X = latest_data[features]

        prediction = model.predict(X)

        predicted_temperature = prediction[0]

        print("\nForecasted Weather " "for Tomorrow")

        print(f"Temperature: " f"{predicted_temperature:.2f}°C")

        return predicted_temperature
