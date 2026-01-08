import numpy as np

import pandas as pd

df=pd.read_csv("covid_large_dataset_50000_rows.csv")

df["new_cases_avg"] = df["new_cases"].rolling(7,min_periods=1).mean()

df['lag1'] = df['new_cases_avg'].shift(1).fillna(0)
df['lag2'] = df['new_cases_avg'].shift(2).fillna(0)
df['lag3'] = df['new_cases_avg'].shift(3).fillna(0)

numerical_columns= df.select_dtypes(include=["int64","float64"]).columns.to_list()

categorical_columns=df.select_dtypes(include=["object"]).columns.to_list()

from sklearn.impute import SimpleImputer

li=SimpleImputer(strategy="median")

df[numerical_columns]= li.fit_transform(df[numerical_columns])

le=SimpleImputer(strategy="most_frequent")

df[categorical_columns] = le.fit_transform(df[categorical_columns])

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df["date"] = label.fit_transform(df["date"])

df["location"] = label.fit_transform(df["location"])

X = df.drop(["new_cases_avg"],axis=1)

y=df["new_cases_avg"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=200,max_depth=15,random_state=42)

rf.fit(X_train,y_train)

y_pred= rf.predict(X_test)

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test,y_pred)

from sklearn.metrics import r2_score

print(r2_score(y_pred,y_test))

import streamlit as st

import matplotlib.pyplot as pt

st.header("Covid-19-Prediction")

st.subheader("Actual vs Predicted")

pt.figure(figsize=(8,4))

sample=500

pt.plot(y_test.values[:sample],label="Actual",linewidth=2)

pt.plot(y_pred[:sample],label="Predicted",linewidth=2)

pt.title(f"Actual vs Predicted for {sample} samples")

pt.legend()

fig=pt.gcf()

st.pyplot(fig)



st.subheader("Predict new Covid Cases")

total_cases = st.number_input("Total Cases", value=1000.0)
total_deaths = st.number_input("Total Deaths", value=20.0)
people_vaccinated = st.number_input("People Vaccinated", value=500.0)

last_new_cases_avg = df['new_cases_avg'].iloc[-1]
last_lag1 = df['lag1'].iloc[-1]
last_lag2 = df['lag2'].iloc[-1]
last_lag3 = df['lag3'].iloc[-1]

#Filling remaining columns with realistic default values
input_dict = {
    "date": 0,                 
    "location": 0,                    
    "total_cases": total_cases,
    "new_cases": 50,            
    "total_deaths": total_deaths,
    "new_deaths": 1,                 
    "people_vaccinated": people_vaccinated,
    "people_fully_vaccinated": 300,  
    "new_tests": 1000,               
    "stringency_index": 60,           
    "population": 1000000,            
    "population_density": 500,      
    "hospital_patients": 20,         
    "icu_patients": 5,              
    "reproduction_rate": 1.1,
    "lag1": last_lag1,
    "lag2": last_lag2,  
    "lag3": last_lag3        
}

input_df=pd.DataFrame([input_dict])

input_scaled= scaler.transform(input_df)

if st.button("Predict New Cases"):

    predicted=rf.predict(input_scaled)

    st.success(f"New predicted cases are : {round(predicted[0],2)}")