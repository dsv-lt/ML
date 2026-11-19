"""
Лабораторная работа №2: бинарный кредитный скоринг (да / нет).
Модели:
1) Логистическая регрессия
2) Деревья решений
3) SVM
4) Бустинг (GradientBoosting, HistGradientBoosting, AdaBoost из sklearn)
5) Ансамблинг (RandomForest, VotingClassifier, StackingClassifier)

Используется только встроенный оффлайн-генератор данных scikit-learn
(без скачивания датасетов из сети и без сторонних библиотек бустинга).
"""

import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Фиксируем все генераторы случайных чисел
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "lab2")
os.makedirs(OUT_DIR, exist_ok=True)


def eval_scoring_model(name, group, model, X_test, y_test):
    """Считает ключевые метрики кредитного скоринга на тестовой выборке."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    gini = 2.0 * auc - 1.0  # коэффициент Джини: линейная связь с ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks = float(np.max(tpr - fpr))  # статистика Колмогорова-Смирнова

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Модель": name,
        "Группа": group,
        "ROC-AUC": auc,
        "Gini": gini,
        "KS": ks,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }
    return metrics, (fpr, tpr, auc), cm


def main():
    print("=" * 72)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: БИНАРНЫЙ КРЕДИТНЫЙ СКОРИНГ (SCIKIT-LEARN)")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 0. Данные и разбиение (встроенный оффлайн-генератор make_classification)
    # ------------------------------------------------------------------
    feature_names = [
        "age",
        "monthly_income",
        "loan_amount",
        "credit_term",
        "debt_to_income",
        "credit_history_len",
        "num_open_loans",
        "num_delays_90d",
        "employment_years",
        "savings_ratio",
        "collateral_score",
        "utilization_rate",
    ]

    # Датасет сгенерирован встроенным make_classification из sklearn без загрузки из сети
    X_raw, y_raw = make_classification(
        n_samples=2500,
        n_features=12,
        n_informative=7,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.74, 0.26],
        flip_y=0.04,
        class_sep=0.85,
        random_state=SEED,
    )

    df = pd.DataFrame(X_raw, columns=feature_names)
    df["default"] = y_raw  # 0 - одобрен (нет дефолта), 1 - отказ (дефолт)

    print("\n--- 0. Краткий EDA ---")
    print(f"Размер датасета: {df.shape[0]} объектов, {len(feature_names)} признаков")
    class_counts = df["default"].value_counts().sort_index()
    print(
        f"Баланс классов: 0 (одобрен) = {class_counts[0]} ({class_counts[0] / len(df):.1%}), "
        f"1 (дефолт) = {class_counts[1]} ({class_counts[1] / len(df):.1%})"
    )

    X = df[feature_names]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"Train: {X_train.shape[0]} строк | Test: {X_test.shape[0]} строк")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = []
    roc_data = {}
    conf_matrices = {}

    # ------------------------------------------------------------------
    # 1. Логистическая регрессия
    # ------------------------------------------------------------------
    print("\n--- 1. Логистическая регрессия ---")
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    logreg_grid = GridSearchCV(
        logreg_pipe,
        param_grid={
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    logreg_grid.fit(X_train, y_train)
    best_logreg = logreg_grid.best_estimator_
    print(f"Лучшие параметры: {logreg_grid.best_params_}, CV ROC-AUC = {logreg_grid.best_score_:.4f}")

    # Веса признаков логистической регрессии
    coefs = best_logreg.named_steps["clf"].coef_[0]
    coef_series = pd.Series(coefs, index=feature_names).sort_values(key=np.abs, ascending=False)
    print("Топ-6 весов признаков (по |w|):")
    for fname, w in coef_series.head(6).items():
        print(f"  {fname:20s}: {w:+.4f}")

    m, r, cm = eval_scoring_model("LogisticRegression", "1. LogReg", best_logreg, X_test, y_test)
    results.append(m)
    roc_data["LogisticRegression"] = r
    conf_matrices["LogisticRegression"] = cm

    # ------------------------------------------------------------------
    # 2. Деревья решений
    # ------------------------------------------------------------------
    print("\n--- 2. Деревья решений ---")
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        param_grid={
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [5, 15, 30],
            "class_weight": [None, "balanced"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    print(f"Лучшие параметры: {dt_grid.best_params_}, CV ROC-AUC = {dt_grid.best_score_:.4f}")

    dt_train_auc = roc_auc_score(y_train, best_dt.predict_proba(X_train)[:, 1])
    dt_test_auc = roc_auc_score(y_test, best_dt.predict_proba(X_test)[:, 1])
    print(f"Контроль переобучения: Train ROC-AUC = {dt_train_auc:.4f}, Test ROC-AUC = {dt_test_auc:.4f}")

    dt_imp = pd.Series(best_dt.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("Топ-5 важных признаков дерева:")
    for fname, imp in dt_imp.head(5).items():
        print(f"  {fname:20s}: {imp:.4f}")

    m, r, cm = eval_scoring_model("DecisionTree", "2. DecisionTree", best_dt, X_test, y_test)
    results.append(m)
    roc_data["DecisionTree"] = r
    conf_matrices["DecisionTree"] = cm

    # ------------------------------------------------------------------
    # 3. SVM (сравнение linear и rbf ядер)
    # ------------------------------------------------------------------
    print("\n--- 3. SVM ---")
    svm_linear_grid = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(SVC(kernel="linear", random_state=SEED), cv=5, ensemble=False)),
        ]),
        param_grid={"clf__estimator__C": [0.1, 1.0, 5.0]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    svm_linear_grid.fit(X_train, y_train)
    best_svm_lin = svm_linear_grid.best_estimator_
    print(f"SVM (linear) лучшие: {svm_linear_grid.best_params_}, CV ROC-AUC = {svm_linear_grid.best_score_:.4f}")

    svm_rbf_grid = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), cv=5, ensemble=False)),
        ]),
        param_grid={
            "clf__estimator__C": [0.1, 1.0, 5.0],
            "clf__estimator__gamma": ["scale", 0.05, 0.1],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    svm_rbf_grid.fit(X_train, y_train)
    best_svm_rbf = svm_rbf_grid.best_estimator_
    print(f"SVM (rbf) лучшие:    {svm_rbf_grid.best_params_}, CV ROC-AUC = {svm_rbf_grid.best_score_:.4f}")

    m_lin, r_lin, cm_lin = eval_scoring_model("SVM (linear)", "3. SVM", best_svm_lin, X_test, y_test)
    m_rbf, r_rbf, cm_rbf = eval_scoring_model("SVM (rbf)", "3. SVM", best_svm_rbf, X_test, y_test)
    results.extend([m_lin, m_rbf])
    roc_data["SVM (linear)"] = r_lin
    roc_data["SVM (rbf)"] = r_rbf
    conf_matrices["SVM (linear)"] = cm_lin
    conf_matrices["SVM (rbf)"] = cm_rbf

    if svm_rbf_grid.best_score_ >= svm_linear_grid.best_score_:
        best_svm = best_svm_rbf
        best_svm_name = "SVM (rbf)"
    else:
        best_svm = best_svm_lin
        best_svm_name = "SVM (linear)"

    # ------------------------------------------------------------------
    # 4. Бустинг (только sklearn.ensemble)
    # ------------------------------------------------------------------
    print("\n--- 4. Бустинг (sklearn.ensemble) ---")
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=SEED),
        param_grid={
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_
    print(f"GradientBoosting:     {gb_grid.best_params_}, CV ROC-AUC = {gb_grid.best_score_:.4f}")

    hgb_grid = GridSearchCV(
        HistGradientBoostingClassifier(random_state=SEED),
        param_grid={
            "max_iter": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, None],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    hgb_grid.fit(X_train, y_train)
    best_hgb = hgb_grid.best_estimator_
    print(f"HistGradientBoosting: {hgb_grid.best_params_}, CV ROC-AUC = {hgb_grid.best_score_:.4f}")

    ada_grid = GridSearchCV(
        AdaBoostClassifier(random_state=SEED),
        param_grid={
            "n_estimators": [50, 150, 250],
            "learning_rate": [0.3, 0.7, 1.0],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    ada_grid.fit(X_train, y_train)
    best_ada = ada_grid.best_estimator_
    print(f"AdaBoost:             {ada_grid.best_params_}, CV ROC-AUC = {ada_grid.best_score_:.4f}")

    for name, mdl in [
        ("GradientBoosting", best_gb),
        ("HistGradientBoosting", best_hgb),
        ("AdaBoost", best_ada),
    ]:
        m, r, cm = eval_scoring_model(name, "4. Boosting", mdl, X_test, y_test)
        results.append(m)
        roc_data[name] = r
        conf_matrices[name] = cm

    # Лучший бустинг по CV
    boost_candidates = [
        ("GradientBoosting", gb_grid.best_score_),
        ("HistGradientBoosting", hgb_grid.best_score_),
        ("AdaBoost", ada_grid.best_score_),
    ]
    best_boost_name = max(boost_candidates, key=lambda x: x[1])[0]

    # ------------------------------------------------------------------
    # 5. Ансамблинг (RandomForest, Voting, Stacking)
    # ------------------------------------------------------------------
    print("\n--- 5. Ансамблинг (sklearn.ensemble) ---")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        param_grid={
            "n_estimators": [150, 300],
            "max_depth": [6, 10, None],
            "min_samples_leaf": [2, 5],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"RandomForest: {rf_grid.best_params_}, CV ROC-AUC = {rf_grid.best_score_:.4f}")

    voting_clf = VotingClassifier(
        estimators=[
            ("logreg", clone(best_logreg)),
            ("svm", clone(best_svm)),
            ("gb", clone(best_gb)),
            ("rf", clone(best_rf)),
        ],
        voting="soft",
        n_jobs=-1,
    )
    voting_clf.fit(X_train, y_train)

    stacking_clf = StackingClassifier(
        estimators=[
            ("logreg", clone(best_logreg)),
            ("dt", clone(best_dt)),
            ("svm", clone(best_svm)),
            ("gb", clone(best_gb)),
            ("rf", clone(best_rf)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=5,
        n_jobs=-1,
    )
    stacking_clf.fit(X_train, y_train)

    for name, mdl in [
        ("RandomForest", best_rf),
        ("Voting (soft)", voting_clf),
        ("Stacking", stacking_clf),
    ]:
        m, r, cm = eval_scoring_model(name, "5. Ensemble", mdl, X_test, y_test)
        results.append(m)
        roc_data[name] = r
        conf_matrices[name] = cm

    # ------------------------------------------------------------------
    # Сводная таблица результатов и графики
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(OUT_DIR, "models_comparison.csv")
    results_df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\n--- Сводная таблица сравнения моделей (отсортировано по ROC-AUC) ---")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Выбираем по одному главному представителю от каждой из 5 групп для графиков
    ens_df = results_df[results_df["Группа"] == "5. Ensemble"]
    best_ens_name = ens_df.iloc[0]["Модель"]

    main_models = [
        "LogisticRegression",
        "DecisionTree",
        best_svm_name,
        best_boost_name,
        best_ens_name,
    ]

    # 1) ROC-кривые представителей всех 5 групп
    plt.figure(figsize=(8, 6))
    for name in main_models:
        fpr, tpr, auc = roc_data[name]
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Случайная модель")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC-кривые моделей кредитного скоринга (5 групп)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    # 2) Столбчатое сравнение ROC-AUC, Gini, F1 по всем моделям
    plot_df = results_df.sort_values("ROC-AUC", ascending=True)
    y_pos = np.arange(len(plot_df))
    h = 0.26

    plt.figure(figsize=(10, 6.5))
    plt.barh(y_pos + h, plot_df["ROC-AUC"], height=h, label="ROC-AUC")
    plt.barh(y_pos, plot_df["Gini"], height=h, label="Gini")
    plt.barh(y_pos - h, plot_df["F1"], height=h, label="F1")
    plt.yticks(y_pos, plot_df["Модель"])
    plt.xlim(0, 1.05)
    plt.xlabel("Значение метрики")
    plt.title("Сравнение моделей скоринга по ROC-AUC, Gini и F1")
    plt.legend(loc="lower right")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metrics_comparison.png"), dpi=150)
    plt.close()

    # 3) Матрицы ошибок для 5 главных моделей
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    for ax, name in zip(axes, main_models):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrices[name],
            display_labels=["0 (одобр.)", "1 (дефолт)"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)
    plt.suptitle("Матрицы ошибок (по одной модели от каждой группы)", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nРезультаты и графики сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()
