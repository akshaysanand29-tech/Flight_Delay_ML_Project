import numpy as np

import pandas as pd

df=pd.read_csv("Airline_Delay_Cause.csv")

df=df.dropna(subset=["arr_delay"])

df["Delayed"]= df["arr_delay"].apply(lambda i: 1 if i>15 else 0 )

df.drop(["arr_delay"],axis=1,inplace=True)

df.drop([
      'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay',
   'arr_cancelled', 'arr_diverted', 'arr_flights', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct',
     'late_aircraft_ct', 'carrier_name', 'airport_name'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["carrier"] = le.fit_transform(df["carrier"])

df["airport"] = le.fit_transform(df["airport"])

important_features = ["carrier", "airport", "arr_del15"]

X = df[important_features]

y = df["Delayed"]

y=df["Delayed"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred,y_test)

from sklearn.metrics import confusion_matrix

matrix=confusion_matrix(y_pred,y_test)

from sklearn.metrics import recall_score

recall= recall_score(y_pred,y_test)

from sklearn.metrics import f1_score

f1_Score=f1_score(y_pred,y_test)

from sklearn.metrics import precision_score

precision=precision_score(y_pred,y_test)

import streamlit as st

import matplotlib.pyplot as plt

st.title("Flight Delay Prediction")

st.subheader("Model Performance Metrics")

st.write(f"Accuracy: {accuracy:.4f}")

st.write(f"Precision (Delayed): {precision:.4f}")

st.write(f"Recall (Delayed): {recall:.4f}")

st.write(f"F1-score (Delayed): {f1_Score:.4f}")

st.subheader("Delayed vs On-Time Flights in Test Set")

counts = pd.Series(y_pred).value_counts().sort_index()

labels = ["On-time", "Delayed"]

plt.bar(labels, counts)

plt.ylabel("Number of Flights")

plt.title("Predicted Flight Status")

st.pyplot(plt)

st.subheader("Predict Flight Delay")

st.sidebar.header("Input Flight Details")

carrier_input = st.sidebar.selectbox("Carrier", sorted(df['carrier'].unique()))
airport_input = st.sidebar.selectbox("Airport", sorted(df['airport'].unique()))
arr_del15_input = st.sidebar.number_input("arr_del15", min_value=0.0, max_value=1.0, value=0.0)

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
"carrier": [carrier_input],
"airport": [airport_input],
"arr_del15": [arr_del15_input]
})

    predict=rf.predict(input_data)[0]

    st.write("The Flight is Delayed" if predict==1 else "The Flight is On Time")




