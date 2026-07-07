import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

df = pd.read_csv("heart.csv")
print(df.head())
df = df.drop_duplicates()

X = df.drop("target", axis = 1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

grid_search= GridSearchCV(
    param_grid=param_grid,
    cv=5,
    scoring= 'accuracy',
    estimator = xgb.XGBClassifier(random_state=42)
    )

grid_search.fit(X_train,y_train)
print("Best params:", grid_search.best_params_)
print("best R2:", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)

print("Tuned Test Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))

importances = best_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(importance_df)