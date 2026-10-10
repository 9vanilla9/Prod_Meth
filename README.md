# Лабораторная работа 1: Введение и воспроизводимость исследований

**Вариант 20.** Синтетический датасет (`make_classification`): 15 числовых и 3 категориальных признака.
Задача — бинарная классификация, целевая переменная `target`.
Кастомный трансформер `FeatureProduct` добавляет признак-произведение `num_2 * num_3`.

## Структура проекта

```
Lab1/
├── data/
│   ├── raw/              # исходные данные (под контролем DVC)
│   └── processed/        # обработанные данные
├── models/               # сохранённые модели (.joblib)
├── notebooks/            # ноутбуки
├── reports/              # метрики и отчёты
├── src/
│   ├── make_dataset.py   # генерация датасета
│   ├── transformers.py   # кастомный трансформер FeatureProduct
│   └── train.py          # обучение и оценка пайплайна
├── .gitignore
├── requirements.txt
└── README.md
```

_Скриншот структуры проекта:_ <!-- вставить скриншот -->

## Шаги

1. Создана структура каталогов, инициализирован Git, добавлен `.gitignore`.
2. Создано виртуальное окружение `.venv`, установлены pandas, scikit-learn, matplotlib, seaborn, dvc, jupyterlab. Зависимости зафиксированы в `requirements.txt`.
3. `src/make_dataset.py` генерирует 2000 объектов:
   - 18 признаков, из них 10 информативных;
   - последние 3 признака разбиты на квантили и превращены в категории `cat_color`, `cat_size`, `cat_group`;
   - в 10% значений колонок `num_0` и `num_1` внесены искусственные пропуски (для тренировки импутации);
   - данные сохранены в `data/raw/synthetic.csv`.
4. Файл данных добавлен под контроль DVC (`dvc init`, `dvc add`). В Git хранится только `synthetic.csv.dvc`.
5. `src/train.py`:
   - разбиение 80/20 (`stratify`, `random_state=42`);
   - пайплайн: `FeatureProduct` → `ColumnTransformer` → `LogisticRegression`:
     - числовые признаки: медиана + `StandardScaler`;
     - категориальные: мода + `OneHotEncoder(handle_unknown='ignore')`;
   - подбор `C` через `GridSearchCV` (5 фолдов, ROC-AUC);
   - модель сохраняется в `models/model.joblib`, метрики — в `reports/metrics.json`.

## Как воспроизвести

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull                    # или: python src/make_dataset.py
python src/train.py
```

## Результаты

| Метрика | Значение |
|---|---|
| Лучшие параметры | — |
| ROC-AUC (CV, 5 фолдов) | — |
| Accuracy (тест) | — |
| ROC-AUC (тест) | — |

_Скриншот метрик:_ <!-- вставить скриншот -->

_Скриншот `dvc status`:_ <!-- вставить скриншот -->

## Выводы

<!-- заполнить после запуска -->
