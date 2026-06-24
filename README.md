# ML Learning Journey 🤖

A documentation of my Machine Learning learning —
including code practice and theory notes.

---

## Train Test Split

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
