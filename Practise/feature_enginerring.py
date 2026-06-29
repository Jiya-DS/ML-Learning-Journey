import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("people.csv")

# check missing values
print(df.isnull().sum())
print(df.describe())

#for outliers
df_original= pd.read_csv("people.csv")
df_original = pd.read_csv("people.csv")
df_original.loc[12] = [210, 200, 25, "Fit"]  # fake outlier row!
print(df_original.describe())

#identifying the outlier value(quantile )
Q1 = df_original["weight_kg"].quantile(0.25)
Q3 = df_original["weight_kg"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Lower bound: {lower}")
print(f"Upper bound: {upper}")
print(f"Outlier value: 200")

#clipping for handling the outliers 
df_original["weight_kg"] = df_original["weight_kg"].clip(lower, upper) 
print(df_original["weight_kg"])

df_original.boxplot()
plt.show()

# Encoding the Categorical column "Label"
df["label"] = df["label"].map({"Fit": 1, "Unfit": 0})
print(df.head())

# New Column 
df["BMI"]=df["weight_kg"]/((df["height_cm"]/100)**2)
print(df[["height_cm","weight_kg","BMI"]].head())

# Feature Scaling
scaler= MinMaxScaler()
df[["height_cm","weight_kg","age","BMI"]]= scaler.fit_transform(df[["height_cm","weight_kg","age","BMI"]])
print(df.describe())

# ✅ Check missing values with isnull().sum()
# ✅ Encode with .map() and get_dummies(){more than 2 category}
# ✅ Scale with MinMaxScaler and StandardScaler
# ✅ Create BMI as a new meaningful feature
# ✅ Detect outliers with boxplot and IQR
# ✅ Handle outliers with .clip()



