import warnings
warnings.filterwarnings("ignore")
#for this Code To Work You Need The Following librarys.
#pip install numpy pandas scikit-learn xgboost matplotlib
#for conda

#conda install numpy pandas scikit-learn matplotlib
#conda install -c conda-forge xgboost
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay, confusion_matrix

from xgboost import XGBClassifier
import xgboost as xgb




#file Dir
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")

train_ds_path = os.path.join(DATA_DIR, "train.csv")
test_ds_path = os.path.join(DATA_DIR, "test.csv")

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def create_features(df):
    df = df.copy()

    df["Cardiac_Workload"] = df["Max HR"] * df["Age"]
    df["Stress_Factor"] = df["ST depression"] * df["Slope of ST"]

    # Protect against divide-by-zero (rare)
    df["Chol_Age_Ratio"] = df["Cholesterol"] / df["Age"].replace(0, np.nan)

    df["HighBP"] = (df["BP"] > 130).astype("int32")
    df["HighCholesterol"] = (df["Cholesterol"] > 240).astype("int32")
    df["totalRiskFactors"] = (df["HighBP"] + df["HighCholesterol"] + df["FBS over 120"]).astype("int32")

    # Downcast to save memory
    int_cols = df.select_dtypes(include=["int64"]).columns
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[int_cols] = df[int_cols].astype("int32")
    df[float_cols] = df[float_cols].astype("float32")

    return df


def cleanData(dataset_path,target='Heart Disease'):
    df = pd.read_csv(dataset_path)
    df = create_features(df)

    df = df.drop(columns=["id"])

    df[target] = df[target].map({"Presence": 1, "Absence": 0}).astype("int32")

    print(df.info())
    return df

def train_data(df,target="Heart Disease"):
    X = df.drop(columns=[target])
    y = df[target]

    # Load test
    test_df_raw = pd.read_csv(test_ds_path)
    test_ids = test_df_raw["id"].copy()
    X_test = create_features(test_df_raw.drop(columns=["id"]))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)

    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    cv_scores = []

    last_model = None

    for f, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        spw = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            max_depth=3,
            n_estimators=1200,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.7,
            gamma=0.1,
            min_child_weight=4,
            reg_lambda=2.0,
            reg_alpha=0,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        proba_val = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = proba_val.astype(np.float32)

        test_preds += model.predict_proba(X_test)[:, 1].astype(np.float32)

        score = roc_auc_score(y_val, proba_val)
        cv_scores.append(score)
        print(f"Fold {f} AUC: {score:.5f}")

        last_model = model

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"Overall CV AUC (oof): {overall_auc:.5f}")
    print(f"Average Fold AUC: {np.mean(cv_scores):.5f}")

    # Save submission
    test_preds /= cv.n_splits
    submission = pd.DataFrame({"id": test_ids, target: test_preds})
    submission_path = os.path.join(BASE_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")
    # OOF ROC curve
    save_roc_curve(
        y_true=y,
        y_proba=oof_preds,
        out_path=os.path.join(PLOTS_DIR, "roc_oof.png"),
        title=f"ROC Curve (OOF) - AUC={overall_auc:.5f}"
    )

    #Confusion matrix on OOF predictions using threshold 0.40
    threshold = 0.40
    y_pred_oof = (oof_preds >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred_oof)
    save_confusion_matrix(
        cm,
        out_path=os.path.join(PLOTS_DIR, f"confusion_matrix_oof_thr_{threshold:.2f}.png"),
        title=f"Confusion Matrix (OOF, thr={threshold:.2f})"
    )

    # Feature importance + one tree from the last fold model
    # (You can also refit a final model on full data if you like)
    if last_model is not None:
        save_feature_importance(
            last_model,
            out_path=os.path.join(PLOTS_DIR, "importance_gain.png"),
            importance_type="gain"
        )
        save_feature_importance(
            last_model,
            out_path=os.path.join(PLOTS_DIR, "importance_weight.png"),
            importance_type="weight"
        )
        save_tree(
            last_model,
            out_path=os.path.join(PLOTS_DIR, "tree_example.png"),
            num_trees=2
        )

    print(f"Saved plots to: {PLOTS_DIR}")


#Plots,Saves To Your Dir
def save_feature_importance(model, out_path: str, importance_type: str = "gain", max_num_features: int = 20):
    #Futcure importance Plot
    plt.figure(figsize=(10, 6), dpi=150)
    ax = xgb.plot_importance(model, importance_type=importance_type, max_num_features=max_num_features)
    ax.set_title(f"Feature Importance ({importance_type})")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_tree(model, out_path: str, num_trees: int = 0):
    #tree generated By The Model
    plt.figure(figsize=(30, 12), dpi=200)
    xgb.plot_tree(model, num_trees=num_trees)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_proba, out_path: str, title: str):
    #ROC Curve
    plt.figure(figsize=(7, 6), dpi=150)
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, out_path: str, title: str = "Confusion Matrix"):
    # heatmap for Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    # Put numbers inside cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
if __name__ == "__main__":
    df_pre = cleanData(train_ds_path)
    train_data(df_pre)