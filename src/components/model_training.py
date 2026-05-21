import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score


class ModelTraining:

    def __init__(self):

        self.processed_data_path = "data/processed/" "weather_processed.csv"

        self.model_path = "models/weather_model.pkl"

    def load_data(self):

        dataframe = pd.read_csv(self.processed_data_path)

        return dataframe

    def train_model(self):

        dataframe = self.load_data()

        # Predict next hour temperature
        dataframe["target_temperature"] = dataframe["temperature_2m"].shift(-1)

        dataframe.dropna(inplace=True)

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

        X = dataframe[features]

        y = dataframe["target_temperature"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        r2 = r2_score(y_test, predictions)

        print(f"MAE: {mae:.2f}")

        print(f"R2 Score: {r2:.2f}")

        return model

    def save_model(self):

        model = self.train_model()

        save_path = Path(self.model_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, save_path)

        print("Model saved successfully!")
