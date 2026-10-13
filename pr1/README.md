# Практическое задание 1: структура DS-проекта, Git и DVC

Учебный проект, в котором настроены изолированное окружение, структура каталогов по мотивам Cookiecutter Data Science и версионирование данных через DVC.

## Что сделано

1. Виртуальное окружение `.venv` на Python 3.12 (`venv`).
2. Структура каталогов по шаблону.
3. Добавлены `.gitignore`, `requirements.txt` (через `pip freeze`) и `README.md`.
4. Скрипт `src/data/make_dataset.py` генерирует синтетический датасет `data/raw/data.csv`: 1000 строк, 7 столбцов, `seed=42`.
5. DVC инициализирован, CSV добавлен под контроль DVC и отправлен в удалённое хранилище (`dvc push`).

## Структура

```
.
├── data/
│   ├── raw/              # исходные данные (data.csv под DVC)
│   ├── processed/        # очищенные данные
│   └── external/         # внешние данные
├── notebooks/            # exploratory analysis
├── src/
│   ├── data/             # загрузка и генерация данных (make_dataset.py)
│   ├── features/         # генерация признаков
│   ├── models/           # обучение моделей
│   └── visualization/    # визуализация
├── models/               # сохранённые модели
├── reports/              # отчёты и графики
├── .dvc/                 # конфигурация DVC
├── requirements.txt
└── README.md
```

## Данные

| Столбец | Описание |
|---|---|
| client_id | идентификатор клиента |
| age | возраст, 18–69 |
| income | доход, около 5% пропусков |
| city | город, около 3% пропусков |
| segment | сегмент A / B / C |
| visits | число визитов (Пуассон, λ=5) |
| churn | целевая переменная 0/1 |

В Git хранится только файл метаданных `data/raw/data.csv.dvc`. Сам CSV лежит в кэше DVC и в удалённом хранилище.

## Запуск

```bash
git clone <repo-url>
cd ProdMeth_1Pr
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Получить данные:

```bash
dvc pull
```

Remote `myremote` настроен на локальную папку (`/Users/egor/dvc_storage/ProdMeth_1Pr`), поэтому `dvc pull` сработает только на машине автора. На другой машине датасет можно воспроизвести скриптом: результат будет тем же, так как зафиксирован `seed`.

```bash
python src/data/make_dataset.py
```

## Основные команды DVC, которые использовались

```bash
dvc init
dvc add data/raw/data.csv
dvc remote add -d myremote ~/dvc_storage/ProdMeth_1Pr
git add data/raw/data.csv.dvc data/raw/.gitignore .dvc/
git commit -m "..."
dvc push
```
