import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

NAME = "Sree Vardhan Reddy"
ROLL_NO = "2022BCS0056"

np.random.seed(42)
n = 1800

df = pd.DataFrame({
    "fixed_acidity": np.random.uniform(4.0, 15.0, n),
    "volatile_acidity": np.random.uniform(0.10, 1.20, n),
    "citric_acid": np.random.uniform(0.00, 1.00, n),
    "residual_sugar": np.random.uniform(1.0, 15.0, n),
    "chlorides": np.random.uniform(0.01, 0.20, n),
    "free_sulfur_dioxide": np.random.uniform(1.0, 70.0, n),
    "total_sulfur_dioxide": np.random.uniform(6.0, 200.0, n),
    "density": np.random.uniform(0.990, 1.004, n),
    "pH": np.random.uniform(2.8, 4.0, n),
    "sulphates": np.random.uniform(0.30, 1.50, n),
    "alcohol": np.random.uniform(8.0, 14.0, n),
})

noise = np.random.normal(0, 0.35, n)

quality_score = (
    5.0
    + 0.25 * (df["alcohol"] - 10.0)
    - 2.0 * (df["volatile_acidity"] - 0.50)
    + 1.5 * (df["citric_acid"] - 0.30)
    + 1.2 * (df["sulphates"] - 0.60)
    - 12.0 * (df["chlorides"] - 0.06)
    + noise
)

df["quality"] = np.clip(np.round(quality_score), 3, 8).astype(int)

X = df.drop(columns=["quality"])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)

joblib.dump(model, "model.pkl")

metrics = {
    "name": NAME,
    "roll_no": ROLL_NO,
    "mse": round(float(mse), 6)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

sample_input = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.50,
    "citric_acid": 0.30,
    "residual_sugar": 5.0,
    "chlorides": 0.06,
    "free_sulfur_dioxide": 20.0,
    "total_sulfur_dioxide": 80.0,
    "density": 0.996,
    "pH": 3.30,
    "sulphates": 0.60,
    "alcohol": 10.0
}

with open("sample_input.json", "w") as f:
    json.dump(sample_input, f, indent=2)

print(f"{ROLL_NO} ---- Training completed")
print(f"Name: {NAME}")
print(f"Roll No: {ROLL_NO}")
print(f"MSE: {mse:.6f}")