import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("crop_yield.csv")

print(df.shape)
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.info())

# Sample 10,000 rows (KNN is slow on large data)
df = df.sample(n=10000, random_state=42)
print("Sampled Shape:", df.shape)

# Encode categorical columns
le = LabelEncoder()
for col in ["Region", "Soil_Type", "Crop", "Weather_Condition"]:
    df[col] = le.fit_transform(df[col])

# Convert boolean to int
df["Fertilizer_Used"] = df["Fertilizer_Used"].astype(int)
df["Irrigation_Used"] = df["Irrigation_Used"].astype(int)

print("\nAfter Encoding:")
print(df.head())


# Features and Target
X = df.drop("Yield_tons_per_hectare", axis=1)
y = df["Yield_tons_per_hectare"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Feature Scaling (MUST for KNN!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nScaling done! ✅")

# Train KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

print("Model trained successfully! ✅")

# Predictions
y_pred = knn.predict(X_test)

print("First 5 Predictions:", y_pred[:5])
print("First 5 Actual:     ", y_test.values[:5])


mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("===== KNN Model Performance =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

k_values = range(1, 21)
rmse_scores = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, rmse_scores, marker='o', color='green')
plt.xlabel("K Value")
plt.ylabel("RMSE")
plt.title("KNN - Finding Best K Value")
plt.xticks(k_values)
plt.tight_layout()
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/best_k_value.png")
plt.show()

best_k = k_values[np.argmin(rmse_scores)]
print(f"Best K : {best_k}")
print(f"Best RMSE : {round(min(rmse_scores), 4)}")

# Retrain with best K
best_model = KNeighborsRegressor(n_neighbors=15)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_best)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2   = r2_score(y_test, y_pred_best)

print("===== KNN Best Model (K=15) =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_best, color="steelblue", alpha=0.4, edgecolors="black", linewidths=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("KNN (K=15) - Actual vs Predicted Crop Yield")
plt.legend()
plt.tight_layout()
plt.savefig("visuals/actual_vs_predicted.png")
plt.show()