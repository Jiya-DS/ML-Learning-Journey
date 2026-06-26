import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

df= pd.read_csv("heart.csv")
print(df.head())
print(df.shape)

X=df.drop("target", axis=1)
y=df["target"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Confusion_matrix:",confusion_matrix(y_test, y_pred))

importance=pd.Series(model.feature_importances_,index=X.columns)
importance.sort_values().plot(kind='barh',color='green')
plt.title("Feature Importance - Random Forest Heart Disease")
plt.tight_layout()
os.makedirs("visuals", exist_ok=True)  # Creates folder if it doesn't exist
plt.savefig("visuals/random_forest.png")
plt.show()