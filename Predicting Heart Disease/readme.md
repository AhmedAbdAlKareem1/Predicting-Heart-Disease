❤️ Heart Disease Prediction using XGBoost
📖 Overview

This project implements a binary classification model to predict the presence of heart disease using XGBoost.

The workflow includes feature engineering, hyperparameter tuning, and robust cross-validation using Out-of-Fold (OOF) evaluation to ensure reliable and unbiased model performance.

🧠 Model Approach

XGBoost (Gradient Boosted Decision Trees)

Stratified K-Fold Cross Validation (5 folds)

Out-of-Fold (OOF) predictions for unbiased evaluation

Hyperparameter tuning using RandomizedSearchCV

Performance evaluated using ROC-AUC and PR-AUC

⚙️ Feature Engineering

The following domain-inspired features were created:

Cardiac_Workload = Max HR × Age

Stress_Factor = ST depression × Slope of ST

Chol_Age_Ratio = Cholesterol / Age

Binary risk indicators:

HighBP

HighCholesterol

Combined total risk factor count

Numeric data types were downcast to reduce memory usage and improve efficiency.

📊 Evaluation Results
Metric	Score
OOF ROC-AUC	~0.955
PR-AUC	~0.948
Accuracy (threshold = 0.40)	~0.886

The model demonstrates strong class discrimination and stable cross-validation performance.

🔁 Validation Strategy

To ensure robust generalization:

Stratified K-Fold (5 folds)

Out-of-Fold predictions used for global ROC-AUC

No data leakage between folds

Final model trained on full dataset after validation

📦 Installation
🔹 Using pip (Recommended)

Install dependencies:

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