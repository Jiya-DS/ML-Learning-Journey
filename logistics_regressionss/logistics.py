import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = {
    "study_hours":[1, 2, 3, 4,5,6,7,8,2,5],
    "previous_marks":[40,50,55,60,65,70,75,80,45,62],
    "passed":[0,0,0,1,1,1,1,1,0,1]
}
new_student=pd.DataFrame({
    "study_hours":[4],
    "previous_marks":[58]
})
df=pd.DataFrame(data)
print(df.head())

X=df[["study_hours","previous_marks"]]
y=df["passed"]

model=LogisticRegression()
model.fit(X,y)
# print("value of x:",X)
# print("value of y:",y)
print("Model Trained Succesfully!")

predictions =model.predict(X)
print("Predictions:",predictions)

print("Accuracy:", accuracy_score(y,predictions))

print("Confusion Matrix:",confusion_matrix(y,predictions))

print("New Student Prediction:",model.predict(new_student))