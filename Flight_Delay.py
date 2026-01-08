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

le_carrier = LabelEncoder()
df["carrier_encoded"] = le_carrier.fit_transform(df["carrier"])
carrier_mapping = dict(zip(le_carrier.classes_, le_carrier.transform(le_carrier.classes_)))

le_airport = LabelEncoder()
df["airport_encoded"] = le_airport.fit_transform(df["airport"])
airport_mapping = dict(zip(le_airport.classes_, le_airport.transform(le_airport.classes_)))

important_features = ["carrier_encoded", "airport_encoded", "arr_del15"]

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

accuracy=accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix

matrix=confusion_matrix(y_test,y_pred)

from sklearn.metrics import recall_score

recall= recall_score(y_test,y_pred)

from sklearn.metrics import f1_score

f1_Score=f1_score(y_test,y_pred)

from sklearn.metrics import precision_score

precision=precision_score(y_test,y_pred)

import streamlit as st

import matplotlib.pyplot as plt

st.title("Flight Delay Prediction")

st.subheader("Model Performance Metrics")

st.subheader("Model Performance Metrics")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision (Delayed): {precision:.4f}")
st.write(f"Recall (Delayed): {recall:.4f}")
st.write(f"F1-score (Delayed): {f1_Score:.4f}")

st.subheader("Predicted Flight Status on Test Set")
counts = pd.Series(y_pred).value_counts().sort_index()
fig2, ax2 = plt.subplots()
ax2.bar(["On-time", "Delayed"], counts, color=["green", "red"])
ax2.set_ylabel("Number of Flights")
ax2.set_title("Predicted Flight Status Distribution")
st.pyplot(fig2)



st.subheader("Predicted Flight Delay")

st.sidebar.header("Input Flight Details")

carrier_input = st.sidebar.selectbox("Carrier", list(carrier_mapping.keys()))
airport_input = st.sidebar.selectbox("Airport", list(airport_mapping.keys()))
historical_delay_input = st.sidebar.selectbox(
    "Historical Delay >15 mins?", 
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes" 
)


if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        "carrier_encoded": [carrier_mapping[carrier_input]],
        "airport_encoded": [airport_mapping[airport_input]],
        "arr_del15": [historical_delay_input]
    })

    predict=rf.predict(input_data)[0]

    if predict == 1:
            st.success("🚨 The Flight is Predicted to be DELAYED!")
    
    else:
            st.info("✅ The Flight is Predicted to be ON TIME.")

    




