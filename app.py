import streamlit as st
import pandas as pd
import plotly.express as px

from src.components.model_prediction import ModelPrediction

# Page settings
st.set_page_config(
    page_title="MLOps Weather Forecasting", page_icon="🌦️", layout="wide"
)

# Title
st.title("🌦️ MLOps Weather Forecasting App")

st.write("Weather Forecast Dashboard")

# Load processed data
dataframe = pd.read_csv("data/processed/weather_processed.csv")

# Prediction
prediction = ModelPrediction()

predicted_temperature = prediction.predict_temperature()

# Forecast Section
st.subheader("Forecasted Weather Metrics " "for Tomorrow")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="🌡️ Temperature", value=f"{predicted_temperature:.2f} °C")

with col2:
    latest_rain = dataframe["rain"].iloc[-1]

    st.metric(label="🌧️ Rain", value=f"{latest_rain:.2f} mm")

# Temperature graph
st.subheader("Temperature Trend")

temperature_fig = px.line(
    dataframe, x="date", y="temperature_2m", title="Temperature Over Time"
)

st.plotly_chart(temperature_fig, use_container_width=True)

# Rain graph
st.subheader("Rain Trend")

rain_fig = px.line(dataframe, x="date", y="rain", title="Rain Over Time")

st.plotly_chart(rain_fig, use_container_width=True)

# Data preview
st.subheader("Processed Weather Data")

st.dataframe(dataframe)
