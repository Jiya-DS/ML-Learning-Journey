import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df= pd.read_csv("heart.csv")
print(df.head())
print(df.shape)
X=df.drop("target", axis=1)
y=df["target"]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Confusion_matrix:",confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(model,feature_names=X.columns,class_names=["No diseases","Diseases"],filled=True)
plt.title("Decision Tree Heart-Diseases")
plt.show()