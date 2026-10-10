"""Обучение пайплайна: FeatureProduct -> ColumnTransformer -> LogisticRegression."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transformers import FeatureProduct

RANDOM_STATE = 42
TARGET = "target"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "synthetic.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


def build_pipeline(num_cols, cat_cols):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    return Pipeline([
        ("product", FeatureProduct("num_2", "num_3", new_col="num_2_x_num_3")),
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    # Новый признак от FeatureProduct тоже числовой — добавляем его в список
    num_cols = X.select_dtypes(include="number").columns.tolist() + ["num_2_x_num_3"]

    pipe = build_pipeline(num_cols, cat_cols)

    param_grid = {"model__C": [0.01, 0.1, 1, 10]}
    search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": search.best_params_,
        "cv_roc_auc": round(search.best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    print("Лучшие параметры:", metrics["best_params"])
    print("ROC-AUC на кросс-валидации:", metrics["cv_roc_auc"])
    print("Accuracy на тесте:", metrics["test_accuracy"])
    print("ROC-AUC на тесте:", metrics["test_roc_auc"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Метрики сохранены: {METRICS_PATH}")


if __name__ == "__main__":
    main()
