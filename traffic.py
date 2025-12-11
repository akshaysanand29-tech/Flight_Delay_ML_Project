import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as pt
import streamlit as st

df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")


df['date_time'] = pd.to_datetime(df['date_time'])

df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month


le = LabelEncoder()
df['weather_main'] = le.fit_transform(df['weather_main'])
df['holiday'] = le.fit_transform(df['holiday'])


features = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 
            'weather_main', 'hour', 'day_of_week', 'month']
X = df[features]
y = df['traffic_volume']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print( r2_score(y_test, y_pred))
print( mean_absolute_error(y_test, y_pred))

st.subheader("Traffic Volume Sample Plot")
sample = df.sample(50)
pt.figure(figsize=(8,4))
pt.scatter(sample['hour'], sample['traffic_volume'], color='blue', label='Actual Traffic')
pt.xlabel("Hour")
pt.ylabel("Traffic Volume")
pt.legend()
pt.tight_layout()
st.pyplot(pt)

st.title("Super Simple Traffic Predictor 🚦")

st.write("Enter traffic info below:")

hour = st.number_input("Hour (0-23)", value=12, step=1)
day = st.number_input("Day of week (0=Mon, 6=Sun)", value=2, step=1)
temp = st.number_input("Temperature (K)", value=290.0, step=1.0)

input_data = pd.DataFrame({'holiday':[0],
    'temp':[temp],
    'rain_1h':[0.0],
    'snow_1h':[0.0],
    'clouds_all':[50],
    'weather_main':[0],
    'hour':[hour],
    'day_of_week':[day],
    'month':[6]})

if st.button("Predict Traffic Volume"):
    input_data = pd.DataFrame({
        'holiday':[0],        # default value
        'temp':[temp],
        'rain_1h':[0.0],      # default value
        'snow_1h':[0.0],      # defau
        'clouds_all':[50],    # default value
        'weather_main':[0],   # default value
        'hour':[hour],
        'day_of_week':[day],
        'month':[6] })
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Traffic Volume")
    st.write(f"{int(prediction)} vehicles/hour")
