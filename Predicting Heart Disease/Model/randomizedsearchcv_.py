import os
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix, \
    classification_report


def tune_xgb(x, y, random_state=42):

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    base_model = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="auc"  #change Based On Your eval Metric
    )

    param_dist = {
        "n_estimators": [300, 500, 800, 1200, 2000],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_child_weight": [1, 2, 4, 6, 8, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5, 1.0],
        "reg_alpha": [0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=True
    )

    search.fit(x, y)
    return search, cv


def oof_eval(best_model, x, y, cv):
    # Out-of-fold probabilities
    oof_proba = cross_val_predict(best_model,x,y,cv=cv,method="predict_proba",n_jobs=-1)[:, 1]

    oof_auc = roc_auc_score(y, oof_proba)
    oof_pr = average_precision_score(y, oof_proba)
    return oof_proba, oof_auc, oof_pr


def threshold_eval(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    cm = confusion_matrix(y_true, pred)
    report = classification_report(y_true, pred)
    return pred, acc, cm, report


def train_data(train_csv_path, target_col="Heart Disease", random_state=42, threshold=0.40):
    df = pd.read_csv(train_csv_path)

    # Map target to 0/1 if it's strings
    if df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"Absence": 0, "Presence": 1})

    x = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print("Target counts:\n", y.value_counts(), "\n")

    # Hyperparameter search (CV)
    search, cv = tune_xgb(x, y, random_state=random_state)
    best_model = search.best_estimator_

    print("="*30)
    print("Best params:", search.best_params_)
    print("Best CV ROC-AUC (from search):", search.best_score_)
    print("="*30)

    # OOF evaluation of the best model
    oof_proba, oof_auc, oof_pr = oof_eval(best_model, x, y, cv)

    print("OOF ROC-AUC:", oof_auc)
    print("OOF PR-AUC:", oof_pr)

    #  Optional: show confusion matrix at a chosen threshold (uses OOF proba)
    pred, acc, cm, report = threshold_eval(y, oof_proba, threshold=threshold)

    print(f"\nOOF Accuracy @ threshold={threshold:.2f}: {acc:.5f}")
    print("OOF Confusion matrix:\n", cm)
    print("\nOOF Classification report:\n", report)

    #Fit final model on ALL data
    best_model.fit(x, y)




if __name__ == "__main__":
    #path to the main Project path,
    main_dir = os.path.dirname(os.path.dirname(__file__))
    #path to the train Dataset
    train_ds_path = os.path.join(main_dir, "Dataset", "train.csv")
    #call The Function.
    train_data(
        train_ds_path,
        target_col="Heart Disease",
        random_state=42,
        threshold=0.40
    )
