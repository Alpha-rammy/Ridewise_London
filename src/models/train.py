import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Load data
# =========================
df = pd.read_csv("data/processed/features.csv")

target = "churned"

X = df.drop(columns=[target, "user_id"], errors="ignore")
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("ridewise-churn-prediction")

# =====================================================
# 1. LOGISTIC REGRESSION
# =====================================================

with mlflow.start_run(run_name="logistic_regression"):

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    lr.fit(X_train, y_train)

    lr_preds = lr.predict_proba(X_test)[:, 1]

    lr_roc = roc_auc_score(y_test, lr_preds)
    lr_pr = average_precision_score(y_test, lr_preds)

    mlflow.log_params({
        "model": "logistic_regression",
        "max_iter": 2000,
        "class_weight": "balanced"
    })

    mlflow.log_metrics({
        "roc_auc": lr_roc,
        "pr_auc": lr_pr
    })

    # FIXED: use lr, not model
    mlflow.sklearn.log_model(lr, name="model")

    print("\nLogistic Regression")
    print("ROC-AUC:", lr_roc)
    print("PR-AUC:", lr_pr)


# =====================================================
# 2. XGBOOST
# =====================================================
with mlflow.start_run(run_name="xgboost"):

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    xgb.fit(X_train, y_train)

    xgb_preds = xgb.predict_proba(X_test)[:, 1]

    xgb_roc = roc_auc_score(y_test, xgb_preds)
    xgb_pr = average_precision_score(y_test, xgb_preds)

    mlflow.log_params({
        "model": "xgboost",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })

    mlflow.log_metrics({
        "roc_auc": xgb_roc,
        "pr_auc": xgb_pr
    })

    mlflow.sklearn.log_model(xgb, name="model")
    
    print("\nXGBoost")
    print("ROC-AUC:", xgb_roc)
    print("PR-AUC:", xgb_pr)