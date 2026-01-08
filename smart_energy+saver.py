import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------
# 1. Sample Dataset
# -------------------------
data = {
    'Units_Last_Month': [250, 320, 400, 280, 500, 350, 450, 380, 420, 300],
    'AC_Usage': [2, 5, 6, 3, 7, 4, 6, 5, 6, 3],
    'Heater_Usage': [0, 0, 2, 1, 3, 1, 2, 1, 2, 0],
    'Fridge_Usage': [24, 24, 24, 24, 24, 24, 24, 24, 24, 24],
    'Lights_Usage': [8, 10, 12, 7, 14, 9, 11, 10, 12, 8],
    'Occupants': [3, 4, 5, 2, 6, 4, 5, 3, 5, 3],
    'Next_Month_Units': [260, 330, 420, 290, 520, 360, 460, 390, 430, 310]
}

df = pd.DataFrame(data)

# -------------------------
# 2. Train Model
# -------------------------
X = df.drop("Next_Month_Units", axis=1)
y = df["Next_Month_Units"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 3. Electricity Tariff
# -------------------------
TARIFFS = {"Residential": 5, "Commercial": 7}
FIXED_CHARGES = 100

# -------------------------
# 4. Streamlit App
# -------------------------
st.title("💡 Smart Energy Saver – Actual vs Predicted Units")

# Plot actual vs predicted for test set
y_pred = model.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel("Actual Units")
ax.set_ylabel("Predicted Units")
ax.set_title("Actual vs Predicted Electricity Units")
st.pyplot(fig)

# -------------------------
# 5. User Inputs
# -------------------------
units_last_month = st.number_input("Units consumed last month (kWh)", min_value=0)
ac_usage = st.slider("AC Usage (hours/day)", 0, 12, 3)
heater_usage = st.slider("Heater Usage (hours/day)", 0, 8, 0)
fridge_usage = st.slider("Fridge Usage (hours/day)", 0, 24, 24)
lights_usage = st.slider("Lights Usage (hours/day)", 0, 16, 8)
occupants = st.number_input("Number of occupants", min_value=1)
tariff_type = st.selectbox("Tariff Type", ["Residential", "Commercial"])

# -------------------------
# 6. Predict & Calculate Bill
# -------------------------
if st.button("Predict Bill"):
    input_df = pd.DataFrame([[units_last_month, ac_usage, heater_usage, fridge_usage, lights_usage, occupants]],
                            columns=X.columns)
    
    predicted_units = model.predict(input_df)[0]
    bill = predicted_units * TARIFFS[tariff_type] + FIXED_CHARGES
    
    st.success(f"🔋 Predicted Electricity Units: {predicted_units:.2f} kWh")
    st.success(f"💰 Estimated Electricity Bill: ₹{bill:.2f}")
