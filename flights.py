import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("flight_delays.csv")

# ----------------------------------------------------
# 1. Convert datetime columns
# ----------------------------------------------------
df['ScheduledDeparture'] = pd.to_datetime(df['ScheduledDeparture'], errors='coerce')
df['ScheduledArrival'] = pd.to_datetime(df['ScheduledArrival'], errors='coerce')

# ----------------------------------------------------
# 2. Extract datetime features
# ----------------------------------------------------
df['DepHour'] = df['ScheduledDeparture'].dt.hour
df['DepMinute'] = df['ScheduledDeparture'].dt.minute
df['ArrHour'] = df['ScheduledArrival'].dt.hour
df['ArrMinute'] = df['ScheduledArrival'].dt.minute

# ----------------------------------------------------
# 3. Drop original datetime columns
# ----------------------------------------------------
df = df.drop(columns=['ScheduledDeparture', 'ScheduledArrival'])

# ----------------------------------------------------
# 4. Identify ALL remaining object columns
# ----------------------------------------------------
object_cols = df.select_dtypes(include=['object']).columns
print("Object columns:", object_cols)

# ----------------------------------------------------
# 5. One-hot encode ALL object columns
# ----------------------------------------------------
df = pd.get_dummies(df, columns=object_cols, drop_first=True)

# ----------------------------------------------------
# 6. Split X and y
# ----------------------------------------------------
y = df["DelayMinutes"]
X = df.drop(columns=["DelayMinutes"])

# ----------------------------------------------------
# 7. Train/test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 8. Scale ONLY numeric columns
# ----------------------------------------------------
numeric_cols = ['DepHour','DepMinute','ArrHour','ArrMinute','Distance']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ----------------------------------------------------
# 9. Train model
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model trained successfully!")
