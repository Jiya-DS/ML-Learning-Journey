import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# Load dataset
df = pd.read_csv('StudentsPerformance.csv')
print(df.duplicated().sum())
# Label Encoding for categorical columns 
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns
# Output: Index(['gender', 'race/ethnicity', ...], dtype='object'
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('math score', axis=1)  
y = df['math score']


param_grid = {'max_depth': [3, 5, 7, 10, None],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)
grid_search.fit(X, y)

print("Best params:", grid_search.best_params_)
print("Best R²:", grid_search.best_score_)



default_tree = DecisionTreeRegressor(random_state=42)
default_scores = cross_val_score(default_tree, X, y, cv=5, scoring='r2')
print("Default Tree R²:", default_scores.mean())
print("Tuned Tree R²:", grid_search.best_score_)
