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
