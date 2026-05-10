import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("dataset/Housing.csv")

print("Dataset Loaded Successfully!\n")

print(df.head())

# ==============================
# ENCODE CATEGORICAL COLUMNS
# ==============================

le = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# ==============================
# FEATURES AND TARGET
# ==============================

X = df.drop("price", axis=1)
y = df["price"]

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# LINEAR REGRESSION MODEL
# ==============================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_score = r2_score(y_test, lr_predictions)

print("\nLinear Regression Accuracy:",
      round(lr_score * 100, 2), "%")

# ==============================
# RANDOM FOREST MODEL
# ==============================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_score = r2_score(y_test, rf_predictions)

print("Random Forest Accuracy:",
      round(rf_score * 100, 2), "%")

# ==============================
# SAVE BEST MODEL
# ==============================

os.makedirs("model", exist_ok=True)

if rf_score > lr_score:

    best_model = rf_model
    best_model_name = "Random Forest"

else:

    best_model = lr_model
    best_model_name = "Linear Regression"

joblib.dump(best_model, "model/model.pkl")

print(f"\nBest Model: {best_model_name}")

print("Best model saved successfully!")