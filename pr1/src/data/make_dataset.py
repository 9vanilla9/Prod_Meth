"""Генерация синтетического датасета клиентов и сохранение в data/raw/data.csv."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "raw" / "data.csv"
N_ROWS = 1000
SEED = 42


def make_dataset(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "client_id": np.arange(1, n_rows + 1),
            "age": rng.integers(18, 70, size=n_rows),
            "income": rng.normal(60_000, 15_000, size=n_rows).round(2),
            "city": rng.choice(["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk"], size=n_rows),
            "segment": rng.choice(["A", "B", "C"], size=n_rows, p=[0.5, 0.3, 0.2]),
            "visits": rng.poisson(5, size=n_rows),
            "churn": rng.integers(0, 2, size=n_rows),
        }
    )

    # немного пропусков, чтобы данные были похожи на реальные
    df.loc[rng.random(n_rows) < 0.05, "income"] = np.nan
    df.loc[rng.random(n_rows) < 0.03, "city"] = np.nan

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено {df.shape[0]} строк, {df.shape[1]} столбцов в {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
