import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
df = pd.read_csv("airline_delay.csv")

# -----------------------------
# 2. HANDLE MISSING VALUES
# -----------------------------
df = df.dropna(subset=["arr_delay"])  # target column

# Fill remaining missing values with 0
df.fillna(0, inplace=True)

# -----------------------------
# 3. DROP COLUMNS NOT KNOWN BEFORE FLIGHT
# (avoid data leakage)
# -----------------------------
df.drop([
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "airport_name",
    "carrier_name"
], axis=1, inplace=True, errors="ignore")

# -----------------------------
# 4. ENCODE CATEGORICAL DATA
# -----------------------------
cat_cols = df.select_dtypes(include=["object"]).columns

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# 5. FEATURES & TARGET
# -----------------------------
X = df.drop("arr_delay", axis=1)
y = df["arr_delay"]

# -----------------------------
# 6. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. MODEL TRAINING
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    
)

model.fit(X_train, y_train)

# -----------------------------
# 8. PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 9. MODEL EVALUATION
# -----------------------------
R2_score = r2_score(y_test,y_pred)
print(R2_score)
print("MAE:", mean_absolute_error(y_test, y_pred))
