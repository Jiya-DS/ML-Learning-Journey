# ML Learning Journey 🤖

A documentation of my Machine Learning learning —
including code practice and theory notes.

---

## 01-Train Test Split

### What is Train-Test Split?

Train-Test Split is a technique used in Machine Learning to divide
a dataset into two parts:

- **Training data** → the model learns from this
- **Testing data** → the model is evaluated on this

We never let the model see testing data during training.
This ensures the model is tested on unseen data — just like a real exam.

### Why do we split data?

If we train and test on the same data, the model simply memorizes
the answers. It will perform well on known data but fail on new data.
This problem is called **overfitting.**

### Key Parameters

| Parameter       | Meaning                                       |
| --------------- | --------------------------------------------- |
| `test_size=0.2` | 20% for testing, 80% for training             |
| `random_state`  | Fixes the shuffle so results are reproducible |

### Key Formula

Training size = total items × (1 - test_size)
Testing size = total items × test_size

### Tools Used

- Python
- Scikit-learn

## 02 — Linear Regression

**Dataset:** Agriculture Crop Yield (Kaggle)  
**Goal:** Predict crop yield from weather and farming features

---

### What is Linear Regression?

Linear Regression is a supervised machine learning algorithm that predicts
a numeric value by finding the best straight line through the data.

The equation it learns:
y = mx + b
y → what we predict (Yield)

m → slope (how much y changes per unit of x)

x → input feature (Rainfall, Temperature etc.)

b → intercept (starting point of the line)

---

### Types of Linear Regression

#### 1. Simple Linear Regression

- Uses **only one feature** to predict the target
- Example: Rainfall → Yield
- Equation: `y = mx + b`

#### 2. Multiple Linear Regression

- Uses **two or more features** to predict the target
- Example: Rainfall + Temperature + Fertilizer → Yield
- Equation: `y = m1x1 + m2x2 + m3x3 + b`

#### Key Difference

|          | Simple        | Multiple             |
| -------- | ------------- | -------------------- |
| Features | 1             | 2 or more            |
| Equation | y = mx + b    | y = m1x1 + m2x2 + b  |
| Graph    | Straight line | Cannot plot directly |
| Our R²   | 0.570         | 0.916                |

---

### How It Works

1. Split data into training and testing sets
2. Model learns the slope and intercept from training data
3. Model predicts yield on unseen test data
4. We measure performance using R² and MSE

---

### Key Metrics

| Metric       | Meaning                                              |
| ------------ | ---------------------------------------------------- |
| **R² Score** | How well features explain yield (1.0 =perfect)       |
| **MSE**      | Average squared error of predictions (lower =better) |

---

### Results

| Model             | Features Used                   | R² Score  |
| ----------------- | ------------------------------- | --------- |
| Simple Regression | Rainfall only                   | 0.570     |
| Multiple v1       | + Temperature + Days to Harvest | 0.578     |
| Multiple v2       | + Fertilizer + Irrigation       | **0.916** |

---

### Key Learnings

- Fertilizer and Irrigation were the most pivotal features
- Adding them pushed R² from 0.57 → 0.91
- Choosing the right features matters more than the algorithm itself
- True/False columns must be encoded to 1/0 before using in a model

### 03-Logistic Regression

## What is Logistic Regression?

Logistic Regression is a classification algorithm that predicts the probability of an outcome being **yes or no (1 or 0)**. Unlike Linear Regression which predicts a number, Logistic Regression predicts a **class**.

---

## Types of Logistic Regression

| Type        | When to Use             | Example             |
| ----------- | ----------------------- | ------------------- |
| Binary      | Only 2 outcomes         | Pass / Fail         |
| Multinomial | 3+ outcomes, no order   | Cat / Dog / Bird    |
| Ordinal     | 3+ outcomes, with order | Low / Medium / High |

> We focused on **Binary Logistic Regression** in this session.

---

## How It Works

- Takes input features and predicts a **probability between 0 and 1**
- If probability >= 0.5 → Class 1 (Yes)
- If probability < 0.5 → Class 0 (No)

```
0.0  ←————————————————→  1.0
NO                        YES
(0%)                    (100%)
```

---

## Difference — Linear vs Logistic Regression

|             | Linear Regression     | Logistic Regression       |
| ----------- | --------------------- | ------------------------- |
| Output      | A number (e.g. 270kg) | A probability (e.g. 0.85) |
| Used for    | Regression            | Classification            |
| Example     | Predict crop yield    | Predict pass/fail         |
| Result type | Continuous            | 0 or 1                    |

---

## Confusion Matrix

A confusion matrix shows exactly where the model was right and where it was wrong.

```
                 Predicted NO    Predicted YES
Actual NO    →       TN               FP
Actual YES   →       FN               TP
```

| Term | Full Name      | Meaning                       |
| ---- | -------------- | ----------------------------- |
| TN   | True Negative  | Correctly predicted No        |
| TP   | True Positive  | Correctly predicted Yes       |
| FP   | False Positive | Predicted Yes but actually No |
| FN   | False Negative | Predicted No but actually Yes |

---

## Projects Built

### Project 1 — Student Pass/Fail Predictor

**File:** `logistics.py`

| Detail           | Info                        |
| ---------------- | --------------------------- |
| Dataset          | Manually created            |
| Rows             | 10                          |
| Features         | study_hours, previous_marks |
| Target           | passed (1 = Pass, 0 = Fail) |
| Train/Test Split | No — dataset too small      |
| Accuracy         | 100%                        |

**Confusion Matrix Result:**

```
[[4 0]
 [0 6]]
```

Model predicted all 10 students correctly.

**New Student Prediction:**

- study_hours = 4, previous_marks = 58 → **Predicted: PASS**

---

### Project 2 — Heart Disease Predictor

**File:** `heart_prediction.py`  
**Dataset:** Heart Disease Dataset — [Kaggle Link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

| Detail           | Info                                 |
| ---------------- | ------------------------------------ |
| Rows             | 1025                                 |
| Columns          | 14                                   |
| Missing Values   | None                                 |
| Target           | target (1 = Disease, 0 = No Disease) |
| Train/Test Split | 80% train / 20% test                 |
| random_state     | 42                                   |
| Accuracy         | 79.5%                                |

**Features Used:**
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

**Confusion Matrix Result:**

```
                 Predicted NO    Predicted YES
Actual NO    →      73               29
Actual YES   →      13               90
```

- 73 patients correctly predicted as healthy
- 90 patients correctly predicted as having heart disease
- 29 false alarms
- 13 missed cases

---

## Key Learnings

- Logistic Regression predicts **class (0 or 1)**, not a number
- Always use train-test split when dataset has **100+ rows**
- Use `max_iter=1000` to avoid ConvergenceWarning on large datasets
- Use `df.drop("target", axis=1)` to select all features cleanly
- Accuracy alone is not enough — always check the **confusion matrix**

---

## What Comes Next

- Decision Tree
- Random Forest
- Model Deployment with Streamlit

# 04-Decision Tree — Heart Disease Predictor

## Concept

A Decision Tree is a supervised ML algorithm that splits data by asking
a series of questions until it reaches a final prediction.
It selects the best split at each step by minimizing **Gini Impurity**.

## Key Terms

- **Root Node** — First question at the top of the tree
- **Branch** — Path taken after a condition is met
- **Leaf Node** — Final prediction, no more splits
- **Gini Impurity** — Measures how mixed a node is (0 = pure, 0.5 = mixed)
- **Pruning** — Limiting tree depth to prevent overfitting
- **max_depth** — Parameter used to control tree size

## Dataset

- Source: Kaggle Heart Disease Dataset
- Rows: 1025, Columns: 14
- Target: 1 = Disease, 0 = No Disease

## Model Results

- Algorithm: DecisionTreeClassifier (max_depth=4)
- Accuracy: 80%

## Confusion Matrix

|                   | Predicted No Disease | Predicted Disease |
| ----------------- | -------------------- | ----------------- |
| Actual No Disease | 71                   | 31                |
| Actual Disease    | 10                   | 93                |

## Key Observation

- Chest pain type (cp) was the dominant feature — used as root node
- 10 actual disease cases were missed (False Negatives)
- In medical predictions, Recall matters as much as Accuracy

## Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 79.5%    |
| Decision Tree       | 80.0%    |

# 05-Random Forest — Heart Disease Predictor

## Concept

Random Forest is a supervised ML algorithm that builds many Decision Trees
and combines their predictions through majority voting.
This group of trees is called an **ensemble.**

## How It Works

1. Build many Decision Trees (n_estimators=100 means 100 trees)
2. Each tree gets a different random sample of data (Bagging)
3. Each tree sees a random subset of features (Random Feature Selection)
4. All trees vote → majority vote = final prediction

## Two Special Tricks

- **Bagging** — each tree trains on a different random sample of data
- **Random Feature Selection** — each tree sees a random subset of features
- Both tricks make trees diverse and reduce overfitting

## Dataset

- Source: Kaggle Heart Disease Dataset
- Rows: 1025, Columns: 14
- Target: 1 = Disease, 0 = No Disease

## Model Results

- Algorithm: RandomForestClassifier (n_estimators=100)
- Accuracy: 98.5%

## Confusion Matrix

|                   | Predicted No Disease | Predicted Disease |
| ----------------- | -------------------- | ----------------- |
| Actual No Disease | 102                  | 0                 |
| Actual Disease    | 3                    | 100               |

## Feature Importance (Top 3)

1. cp — chest pain type (most dominant feature)
2. ca — number of major vessels
3. thalach — maximum heart rate achieved

## Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 79.5%    |
| Decision Tree       | 80.0%    |
| Random Forest       | 98.5%    |

## Key Observation

- Random Forest improved accuracy by 18.5% over Decision Tree
- Missed only 3 actual disease cases vs 10 in Decision Tree
- Random Forest corroborates that chest pain (cp) is the
  most important feature for heart disease prediction
