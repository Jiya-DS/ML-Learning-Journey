import pandas as pd

df = pd.read_csv("customer_churn_prediction_dataset.csv")

df = df.drop(columns=["Churn","customerID"])

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.shape)