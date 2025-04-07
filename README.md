# 🌳 Decision Tree Classifier from Scratch (Iris Dataset)

This repository contains a simple, educational implementation of a **Decision Tree Classifier** built entirely from **scratch using NumPy** — no machine learning libraries used (except for loading and splitting the dataset with scikit-learn).

The classifier is trained on the classic **Iris dataset**, and uses **Entropy** and **Information Gain** to recursively build decision rules for classifying iris flowers.

---

## 📌 Features

- ✅ Entropy and Information Gain calculation  
- ✅ Threshold-based splitting for continuous features  
- ✅ Recursive tree building with early stopping (pure node or max depth)  
- ✅ Predictions for single and multiple samples  
- ✅ Accuracy evaluation on test set  
- ✅ Only uses NumPy (besides dataset loading)

---

## 🧪 Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is used, which includes:

- 150 samples of iris flowers  
- 4 numerical features:  
  - Sepal length  
  - Sepal width  
  - Petal length  
  - Petal width  
- 3 target classes:  
  - Setosa (0)  
  - Versicolor (1)  
  - Virginica (2)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/decision-tree-from-scratch.git
cd decision-tree-from-scratch
