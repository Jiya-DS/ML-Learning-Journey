import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df=pd.read_csv("people.csv")

# df.to_csv('Practise/people.csv', index=False)
X=df.drop("label",axis=1)
y=df["label"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

print("Model Trained Successfully!")

y_pred= knn.predict(X_test)

print("First 5 Prediction:",y_pred[:3])
print("First 5 Actual:",y_test.values[:3])
print("Accuracy:",accuracy_score(y_test,y_pred))


