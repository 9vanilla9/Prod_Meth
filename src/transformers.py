"""Кастомные трансформеры для пайплайна."""
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProduct(BaseEstimator, TransformerMixin):
    """Добавляет новый признак — произведение двух числовых колонок."""

    def __init__(self, col_a, col_b, new_col=None):
        self.col_a = col_a
        self.col_b = col_b
        self.new_col = new_col

    def fit(self, X, y=None):
        # Учить нечего: произведение считается одинаково для train и test
        return self

    def transform(self, X):
        X = X.copy()
        name = self.new_col or f"{self.col_a}_x_{self.col_b}"
        X[name] = X[self.col_a] * X[self.col_b]
        return X
