import openmeteo_requests
import pandas as pd
import requests_cache

from retry_requests import retry
from pathlib import Path


class DataIngestion:

    def __init__(self):

        self.url = "https://archive-api.open-meteo.com/v1/archive"

        self.params = {
            "latitude": 8.5686,
            "longitude": 76.8731,
            "start_date": "2026-05-16",
            "end_date": "2026-05-20",
            "hourly": [
                "temperature_2m",
                "rain",
                "relative_humidity_2m",
                "dew_point_2m",
                "wind_speed_10m",
                "weather_code",
            ],
            "timezone": "auto",
        }

    def fetch_weather_data(self):

        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)

        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

        openmeteo = openmeteo_requests.Client(session=retry_session)

        responses = openmeteo.weather_api(self.url, params=self.params)

        response = responses[0]

        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ).tz_convert(response.Timezone().decode())
        }

        hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()

        hourly_data["rain"] = hourly.Variables(1).ValuesAsNumpy()

        hourly_data["relative_humidity_2m"] = hourly.Variables(2).ValuesAsNumpy()

        hourly_data["dew_point_2m"] = hourly.Variables(3).ValuesAsNumpy()

        hourly_data["wind_speed_10m"] = hourly.Variables(4).ValuesAsNumpy()

        hourly_data["weather_code"] = hourly.Variables(5).ValuesAsNumpy()

        dataframe = pd.DataFrame(data=hourly_data)

        return dataframe

    def save_data(self):

        dataframe = self.fetch_weather_data()

        save_path = Path("data/raw/weather_data.csv")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(save_path, index=False)

        print("Weather data saved successfully!")
