import pandas as pd
from pathlib import Path


class DataPreprocessing:

    def __init__(self):

        self.raw_data_path = "data/raw/weather_data.csv"

        self.processed_data_path = "data/processed/" "weather_processed.csv"

    def load_data(self):

        dataframe = pd.read_csv(self.raw_data_path)

        return dataframe

    def preprocess_data(self):

        dataframe = self.load_data()

        # Convert date column
        dataframe["date"] = pd.to_datetime(dataframe["date"])

        # Remove duplicates
        dataframe.drop_duplicates(inplace=True)

        # Handle missing values
        dataframe = dataframe.ffill()

        # Create time features
        dataframe["year"] = dataframe["date"].dt.year

        dataframe["month"] = dataframe["date"].dt.month

        dataframe["day"] = dataframe["date"].dt.day

        dataframe["hour"] = dataframe["date"].dt.hour

        dataframe["day_of_week"] = dataframe["date"].dt.dayofweek

        # Lag feature
        dataframe["previous_temperature"] = dataframe["temperature_2m"].shift(1)

        # Rolling average
        dataframe["temperature_rolling_mean"] = (
            dataframe["temperature_2m"].rolling(window=3).mean()
        )

        # Remove NaN from lag/rolling
        dataframe.dropna(inplace=True)

        return dataframe

    def save_processed_data(self):

        dataframe = self.preprocess_data()

        save_path = Path(self.processed_data_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(save_path, index=False)

        print("Processed data saved " "successfully!")
