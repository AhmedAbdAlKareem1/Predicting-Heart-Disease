# ❤️ Heart Disease Prediction using XGBoost

## 📖 Overview

This project implements a binary classification model to predict the presence of heart disease using **XGBoost**.

The workflow includes:

- Feature engineering
- Hyperparameter tuning
- Stratified K-Fold cross-validation
- Out-of-Fold (OOF) evaluation

The objective is to build a reliable and reproducible machine learning pipeline for structured medical data.

---

## 🧠 Model Approach

- **XGBoost** (Gradient Boosted Decision Trees)
- **Stratified K-Fold Cross Validation (5 folds)**
- **Out-of-Fold (OOF) predictions** for unbiased evaluation
- **Hyperparameter tuning** using `RandomizedSearchCV`
- Evaluation using **ROC-AUC** and **PR-AUC**

---

## ⚙️ Feature Engineering

The following domain-inspired features were created:

- `Cardiac_Workload = Max HR × Age`
- `Stress_Factor = ST depression × Slope of ST`
- `Chol_Age_Ratio = Cholesterol / Age`
- Binary risk indicators:
  - `HighBP`
  - `HighCholesterol`
- Combined total risk factor count

Numeric data types were downcast to improve memory efficiency.

---

## 📊 Evaluation Results

| Metric | Score |
|--------|--------|
| OOF ROC-AUC | ~0.955 |
| PR-AUC | ~0.948 |
| accuracy| ~0.9536 |

The model demonstrates strong class discrimination and stable cross-validation performance.

---

## 🔁 Validation Strategy

To ensure robust generalization:

- Stratified K-Fold (5 folds)
- Out-of-Fold predictions used for global ROC-AUC
- No data leakage between folds
- Final model trained on the full dataset after validation

---

## 📂 Project Structure


Predicting-Heart-Disease/
│
├── DataSet/
│ ├── train.csv
│ └── test.csv
│
├── Model/
│ ├── Model.py
│ └── randomizedsearchcv_.py
│
├── jypter NoteBook/
│ └── NoteBook File.ipynb
│
├── plots/
│
├── submission.csv
└── README.md


---

## 📦 Installation

### 🔹 Using pip (Recommended)

```bash
pip install -r requirements.txt

Or manually:

pip install numpy pandas scikit-learn xgboost matplotlib
🔹 Using Conda
conda install numpy pandas scikit-learn matplotlib
conda install -c conda-forge xgboost
🚀 How to Run

From the Model directory:

python Model.py

For hyperparameter tuning:

python randomizedsearchcv_.py
📌 Notes

The dataset is not included in this repository.

Results may vary depending on dataset version and random seed.

This project follows a reproducible machine learning pipeline structure.

📜 License

This project is open-source and available under the MIT License.


---

# 🎯 What Changed

- Fixed Markdown formatting
- Proper headers
- Proper code blocks
- Clean spacing
- Professional structure
- Removed extra meta text

---
