# Лабораторные работы по алгоритмам машинного обучения

Репозиторий содержит код, сгенерированные таблицы/графики (`outputs/`) и подробные отчёты (`reports/`) для трёх лабораторных работ по ML. Все скрипты используют только встроенные оффлайн-датасеты `scikit-learn` (без скачивания данных из сети).

## Навигация по работам

| Лабораторная | Код | Отчёт | Артефакты |
|---|---|---|---|
| **№1. Линейная регрессия (PyTorch, Feature Engineering, 4 реализации МНК)** | [`lab1_linear_regression.py`](lab1_linear_regression.py) | [`reports/lab1_report.md`](reports/lab1_report.md) | [`outputs/lab1/`](outputs/lab1/) |
| **№2. Бинарный кредитный скоринг (LogReg, Tree, SVM, Boosting, Ensemble)** | [`lab2_credit_scoring.py`](lab2_credit_scoring.py) | [`reports/lab2_report.md`](reports/lab2_report.md) | [`outputs/lab2/`](outputs/lab2/) |
| **№3. Кластеризация (K-means, GMM) до и после PCA, t-SNE и автоэнкодера (PyTorch)** | [`lab3_clustering_dim_red.py`](lab3_clustering_dim_red.py) | [`reports/lab3_report.md`](reports/lab3_report.md) | [`outputs/lab3/`](outputs/lab3/) |

## Установка и запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Опционально для Linux только на CPU:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Запуск лабораторных работ:

```bash
python lab1_linear_regression.py
python lab2_credit_scoring.py
python lab3_clustering_dim_red.py
```

## Основные результаты

- **Лабораторная №1:** на `load_diabetes` комбинация `SplineTransformer(knots=3)` + отбор `RFE (Ridge, alpha=30.0)` снизила `val-RMSE` с `50.0820` (baseline) до `49.4379`; реализации МНК (`Normal Equations`, `SVD`, `SGD` на PyTorch) сошлись со `sklearn` (`Max |w - w_sklearn| <= 0.0056`).
- **Лабораторная №2:** на задаче бинарного скоринга (`2500 × 12`) лидерами стали `HistGradientBoosting` (`ROC-AUC = 0.9755`, `Gini = 0.9510`), `SVM (rbf)` (`0.9755`) и `Stacking` (`0.9754`, `KS = 0.8798`, `F1 = 0.8727`).
- **Лабораторная №3:** на `load_digits` (`1797 × 64` признаков) понижение размерности через `t-SNE (2D)` повысило `ARI` для `GMM` с `0.3701` (`Raw 64D`) до `0.8344` (`+0.4643`), а сжатие до `10D` (`PCA` и автоэнкодер PyTorch) улучшило `GMM` до `0.5435` и `0.4708` соответственно.
