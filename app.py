import streamlit as st
import pandas as pd
import plotly.express as px

from src.components.model_prediction import ModelPrediction

st.set_page_config(page_title="Weather Forecasting", layout="wide")

st.title("🌦️ MLOps Weather Forecasting App")

# Load processed data
dataframe = pd.read_csv("data/processed/weather_processed.csv")

# Prediction
prediction = ModelPrediction()

temperature = prediction.predict_temperature()

# Metric Card
st.subheader("Forecasted Weather " "for Tomorrow")

col1, col2 = st.columns(2)

with col1:
    st.metric("🌡️ Temperature", f"{temperature:.2f} °C")

with col2:
    st.metric("🌧️ Rain", f"{dataframe['rain'].iloc[-1]:.2f} mm")

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

# Data table
st.subheader("Processed Weather Data")

st.dataframe(dataframe)
