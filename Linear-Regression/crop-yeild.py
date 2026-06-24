from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('crop_yield.csv')

df = df.sample(n=10000, random_state=42)
print(df.shape)
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Encoding True/False to 1/0
df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)

X = df[['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest','Fertilizer_Used','Irrigation_Used']]
y = df['Yield_tons_per_hectare']

X_train,X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)
print("Slope (m):",model.coef_)
print("Intercept (b):",model.intercept_)

y_pred = model.predict(X_test)
print("R2 Score:",r2_score(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))

# plt.scatter(X_test,y_test, color='blue', alpha=0.2, label='Actual')
# plt.plot(X_test,y_pred, color='red', linewidth=2, label='Predicted Line')
# plt.xlabel("Rainfall(mm)")
# plt.ylabel("Yield(tons/hectare")
# plt.title("Simple linear regression:Rainfall vs yield")
# plt.legend()
# plt.savefig("rainfall_vs_yield.png")
# plt.show()
plt.scatter(y_test,y_pred,alpha=0.2, label='Actual')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Multiple-Linear Regression:Actual vs Predicted")
plt.legend()
plt.savefig("Actual_vs_predicted.png")
plt.show()