import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


df = pd.read_csv("StudentsPerformance.csv")

encoder = LabelEncoder()

categorical_column =["gender","race/ethnicity","test preparation course","parental level of education","lunch"]
for i in categorical_column:
    df[i] = encoder.fit_transform(df[i])
    print(df[i])
# print(df.isnull().sum())
# print(df.describe())
print(df.head())
# print(df.shape)
X =df.drop("math score", axis=1)
y=df["math score"]
model=LinearRegression()
score = cross_val_score(model, X,y, cv= 5)
print(score)
print(score.mean())