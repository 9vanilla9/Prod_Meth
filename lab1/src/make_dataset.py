"""Генерация синтетического датасета (вариант 20) и сохранение в data/raw."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42
N_SAMPLES = 2000
MISSING_SHARE = 0.10
MISSING_COLS = ["num_0", "num_1"]

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "synthetic.csv"


def main():
    # 18 числовых признаков: 15 оставим числовыми, 3 превратим в категории
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=18,
        n_informative=10,
        n_redundant=4,
        random_state=RANDOM_STATE,
    )

    df = pd.DataFrame(X[:, :15], columns=[f"num_{i}" for i in range(15)])

    # Категории получаем разбиением числовых колонок на квантили,
    # поэтому они тоже связаны с целевой переменной
    df["cat_color"] = pd.qcut(X[:, 15], q=3, labels=["red", "green", "blue"]).astype(str)
    df["cat_size"] = pd.qcut(X[:, 16], q=4, labels=["S", "M", "L", "XL"]).astype(str)
    df["cat_group"] = pd.qcut(X[:, 17], q=3, labels=["A", "B", "C"]).astype(str)
    df["target"] = y

    # Искусственные пропуски: 10% значений в двух числовых колонках
    rng = np.random.default_rng(RANDOM_STATE)
    n_missing = int(N_SAMPLES * MISSING_SHARE)
    for col in MISSING_COLS:
        idx = rng.choice(N_SAMPLES, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Сохранено: {OUT_PATH}")
    print(f"Размер: {df.shape}")
    print("Пропуски по колонкам:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
