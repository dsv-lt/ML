"""
Лабораторная работа №1 (по слайду «Лабораторная №2: соревнование по линейной регрессии (PyTorch)»).
Только линейные методы (без нейросетей, бустингов и случайного леса).
Цель: минимальный val/test RMSE, логирование MAE и R^2.
Этапы:
0. Подготовка (seed, train/val/test, обучение препроцессинга только на train)
1. Краткий EDA
2. Препроцессинг (пропуски, категориальные, масштабирование, выбросы)
3. Feature Engineering (Log1p, Sqrt/Square, Box-Cox, Yeo-Johnson, полиномы + регуляризация,
   B-сплайны, групповые агрегаты, отбор признаков; таблица трансформаций)
4. Четыре реализации МНК (Normal Equations, SVD, SGD на PyTorch, sklearn для сравнения)

Используется встроенный оффлайн-датасет sklearn.datasets.load_diabetes(scaled=False).
"""

import os
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.datasets import load_diabetes
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    SplineTransformer,
    StandardScaler,
)


# Фиксация всех seed'ов
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "lab1")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.01, 0.1, 1.0, 5.0, 10.0, 30.0, 100.0, 300.0]


def calc_metrics(y_true, y_pred):
    """Считает (RMSE, MAE, R^2)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def eval_linear_ridge(X_tr, y_tr, X_va, y_va, alphas=ALPHAS):
    """Подбирает alpha для Ridge на val и возвращает лучшую модель, alpha и метрики на val."""
    best_alpha = alphas[0]
    best_rmse = float("inf")
    best_model = None
    best_metrics = None

    for a in alphas:
        mdl = Ridge(alpha=a, random_state=SEED)
        mdl.fit(X_tr, y_tr)
        pred_va = mdl.predict(X_va)
        rmse, mae, r2 = calc_metrics(y_va, pred_va)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = a
            best_model = mdl
            best_metrics = (rmse, mae, r2)

    return best_model, best_alpha, best_metrics


def prepare_imputed_splits(df_tr, df_va, df_te, num_cols, cat_cols):
    """Обучает SimpleImputer (строго на train) для числовых и категориальных столбцов."""
    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")

    X_tr_num = num_imp.fit_transform(df_tr[num_cols])
    X_va_num = num_imp.transform(df_va[num_cols])
    X_te_num = num_imp.transform(df_te[num_cols])

    X_tr_cat = cat_imp.fit_transform(df_tr[cat_cols])
    X_va_cat = cat_imp.transform(df_va[cat_cols])
    X_te_cat = cat_imp.transform(df_te[cat_cols])

    df_tr_cat = pd.DataFrame(X_tr_cat, columns=cat_cols, index=df_tr.index)
    df_va_cat = pd.DataFrame(X_va_cat, columns=cat_cols, index=df_va.index)
    df_te_cat = pd.DataFrame(X_te_cat, columns=cat_cols, index=df_te.index)

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    C_tr = ohe.fit_transform(df_tr_cat)
    C_va = ohe.transform(df_va_cat)
    C_te = ohe.transform(df_te_cat)
    cat_names = list(ohe.get_feature_names_out(cat_cols))

    return (
        (X_tr_num, X_va_num, X_te_num),
        (C_tr, C_va, C_te),
        (df_tr_cat, df_va_cat, df_te_cat),
        cat_names,
    )


def scale_train_val_test(X_tr, X_va, X_te):
    """Масштабирует признаки StandardScaler (fit только на train)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)


def clip_iqr_splits(X_tr, X_va, X_te):
    """Вычисляет границы Тьюки [Q1 - 1.5*IQR, Q3 + 1.5*IQR] по train и обрезает выбросы."""
    q1 = np.percentile(X_tr, 25, axis=0)
    q3 = np.percentile(X_tr, 75, axis=0)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    n_clipped_tr = int(np.sum((X_tr < low) | (X_tr > high)))
    n_clipped_va = int(np.sum((X_va < low) | (X_va > high)))
    return (
        np.clip(X_tr, low, high),
        np.clip(X_va, low, high),
        np.clip(X_te, low, high),
        n_clipped_tr,
        n_clipped_va,
    )


def make_group_aggregates(X_tr_num, X_va_num, X_te_num, df_tr_cat, df_va_cat, df_te_cat, num_cols, agg_cols):
    """Считает mean/median/std по категориям строго на train и присоединяет к train/val/test."""
    tr_df = pd.DataFrame(X_tr_num, columns=num_cols, index=df_tr_cat.index)
    va_df = pd.DataFrame(X_va_num, columns=num_cols, index=df_va_cat.index)
    te_df = pd.DataFrame(X_te_num, columns=num_cols, index=df_te_cat.index)

    for c in df_tr_cat.columns:
        tr_df[c] = df_tr_cat[c].values
        va_df[c] = df_va_cat[c].values
        te_df[c] = df_te_cat[c].values

    tr_feats, va_feats, te_feats, names = [], [], [], []
    for cat_col in df_tr_cat.columns:
        stats = tr_df.groupby(cat_col)[agg_cols].agg(["mean", "median", "std"])
        stats.columns = [f"{cat_col}_{col}_{stat}" for col, stat in stats.columns]
        # Глобальные статистики train на случай редких/неизвестных категорий или std=NaN
        global_means = tr_df[agg_cols].mean()
        for col in agg_cols:
            stats[f"{cat_col}_{col}_mean"] = stats[f"{cat_col}_{col}_mean"].fillna(global_means[col])
            stats[f"{cat_col}_{col}_median"] = stats[f"{cat_col}_{col}_median"].fillna(tr_df[col].median())
            stats[f"{cat_col}_{col}_std"] = stats[f"{cat_col}_{col}_std"].fillna(tr_df[col].std())

        m_tr = tr_df[[cat_col]].merge(stats, on=cat_col, how="left").drop(columns=[cat_col])
        m_va = va_df[[cat_col]].merge(stats, on=cat_col, how="left").drop(columns=[cat_col])
        m_te = te_df[[cat_col]].merge(stats, on=cat_col, how="left").drop(columns=[cat_col])

        for col_name in stats.columns:
            fill_val = float(stats[col_name].mean())
            m_va[col_name] = m_va[col_name].fillna(fill_val)
            m_te[col_name] = m_te[col_name].fillna(fill_val)

        tr_feats.append(m_tr.values)
        va_feats.append(m_va.values)
        te_feats.append(m_te.values)
        names.extend(list(stats.columns))

    return (
        np.hstack(tr_feats),
        np.hstack(va_feats),
        np.hstack(te_feats),
        names,
    )


# ----------------------------------------------------------------------
# Четыре реализации МНК (раздел 4)
# ----------------------------------------------------------------------
def fit_normal_equations(X, y, alpha=0.0):
    """МНК через нормальные уравнения: (X_b^T X_b + alpha * D) theta = X_b^T y."""
    n = X.shape[0]
    X_b = np.column_stack([np.ones(n), X])
    d_full = X_b.shape[1]
    reg = np.eye(d_full)
    reg[0, 0] = 0.0  # свободный член b не штрафуем
    A = X_b.T @ X_b + alpha * reg
    rhs = X_b.T @ y
    try:
        theta = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        theta, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    return theta[1:], float(theta[0])


def fit_svd(X, y, alpha=0.0):
    """МНК / Ridge через SVD-разложение центрированной матрицы X_c = U S V^T."""
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    X_c = X - x_mean
    y_c = y - y_mean

    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    if alpha == 0.0:
        cutoff = 1e-10 * s[0] if len(s) > 0 else 0.0
        inv_s = np.where(s > cutoff, 1.0 / s, 0.0)
        w = Vt.T @ (inv_s * (U.T @ y_c))
    else:
        factors = s / (s ** 2 + alpha)
        w = Vt.T @ (factors * (U.T @ y_c))

    b = y_mean - float(x_mean @ w)
    return w, b


def fit_sgd_pytorch(
    X_train,
    y_train,
    X_val,
    y_val,
    alpha=0.0,
    lr=0.02,
    epochs=2000,
    batch_size=64,
):
    """МНК / Ridge через мини-батч SGD в PyTorch (точная эквивалентность целевой функции Ridge)."""
    torch.manual_seed(SEED)
    n_train, d = X_train.shape

    X_tr_t = torch.tensor(X_train, dtype=torch.float64)
    y_tr_t = torch.tensor(y_train, dtype=torch.float64)
    X_va_t = torch.tensor(X_val, dtype=torch.float64)
    y_va_t = torch.tensor(y_val, dtype=torch.float64)

    w = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(y_train.mean()), dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.SGD([w, b], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    gen = torch.Generator().manual_seed(SEED)

    train_hist = []
    val_hist = []

    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=gen)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]

            optimizer.zero_grad()
            pred = xb @ w + b
            # Делим на фиксированный batch_size, а L2-штраф масштабируем на len(idx)/n_train,
            # чтобы сумма по всем батчам эпохи в точности совпадала с функционалом Ridge
            loss = torch.sum((pred - yb) ** 2) / batch_size
            if alpha > 0.0:
                loss = loss + (alpha * len(idx) / (n_train * batch_size)) * torch.sum(w ** 2)
            loss.backward()
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            tr_rmse = torch.sqrt(torch.mean((X_tr_t @ w + b - y_tr_t) ** 2)).item()
            va_rmse = torch.sqrt(torch.mean((X_va_t @ w + b - y_va_t) ** 2)).item()
            train_hist.append(tr_rmse)
            val_hist.append(va_rmse)

    return w.detach().numpy(), float(b.detach().item()), train_hist, val_hist


def main():
    print("=" * 76)
    print("ЛАБОРАТОРНАЯ РАБОТА №1: ЛИНЕЙНАЯ РЕГРЕССИЯ, FEATURE ENGINEERING И МНК")
    print("=" * 76)

    # ------------------------------------------------------------------
    # 0. Подготовка
    # ------------------------------------------------------------------
    data = load_diabetes(scaled=False, as_frame=True)
    df = data.frame.copy()
    target_col = "target"

    # Детерминированные категориальные признаки построчно (без утечки между объектами)
    df["sex_cat"] = np.where(df["sex"] == 1.0, "F", "M")
    df["bmi_cat"] = np.where(
        df["bmi"] < 25.0,
        "normal",
        np.where(df["bmi"] < 30.0, "overweight", "obese"),
    )
    df = df.drop(columns=["sex"])

    num_cols = ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    cat_cols = ["sex_cat", "bmi_cat"]

    # Во встроенном load_diabetes нет пропусков: добавим 3% случайных NaN в числовые
    # признаки (таргет не трогаем), чтобы полноценно проверить этап импьютации.
    rng = np.random.default_rng(SEED)
    mask_nan = rng.random((len(df), len(num_cols))) < 0.03
    num_vals = df[num_cols].to_numpy(dtype=float)
    num_vals[mask_nan] = np.nan
    df[num_cols] = num_vals

    # Разбиение train / val / test = 60% / 20% / 20%
    df_train, df_temp = train_test_split(df, test_size=0.40, random_state=SEED)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=SEED)

    y_train = df_train[target_col].to_numpy(dtype=float)
    y_val = df_val[target_col].to_numpy(dtype=float)
    y_test = df_test[target_col].to_numpy(dtype=float)

    print("\n--- 0. Подготовка ---")
    print(f"Размеры: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    print(f"Числовые ({len(num_cols)}): {num_cols}")
    print(f"Категориальные ({len(cat_cols)}): {cat_cols}")

    # ------------------------------------------------------------------
    # 1. Краткий EDA (строго по train)
    # ------------------------------------------------------------------
    print("\n--- 1. Краткий EDA ---")
    missing_counts = df.drop(columns=[target_col]).isna().sum()
    print("Пропуски по столбцам (весь датасет):")
    print(missing_counts[missing_counts > 0].to_string())

    print(
        f"Таргет на train: mean={y_train.mean():.2f}, std={y_train.std():.2f}, "
        f"min={y_train.min():.1f}, median={np.median(y_train):.1f}, max={y_train.max():.1f}"
    )

    corrs = df_train[num_cols].corrwith(df_train[target_col]).sort_values(key=np.abs, ascending=False)
    print("Корреляция числовых признаков с target (train):")
    for col, cval in corrs.items():
        print(f"  {col:5s}: {cval:+.4f}")

    # График EDA
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(y_train, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Распределение таргета (train)")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("Число объектов")
    axes[0].grid(alpha=0.25)

    corrs_sorted = corrs.sort_values()
    colors = ["tomato" if v < 0 else "seagreen" for v in corrs_sorted.values]
    axes[1].barh(corrs_sorted.index, corrs_sorted.values, color=colors)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("Корреляция Пирсона с target (train)")
    axes[1].set_xlabel("r")
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eda.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 2. Препроцессинг и Baseline
    # ------------------------------------------------------------------
    print("\n--- 2. Препроцессинг и Baseline ---")
    (X_tr_num, X_va_num, X_te_num), (C_tr, C_va, C_te), (df_tr_cat, df_va_cat, df_te_cat), cat_names = (
        prepare_imputed_splits(df_train, df_val, df_test, num_cols, cat_cols)
    )

    # Масштабирование базовых числовых признаков
    S_tr, S_va, S_te = scale_train_val_test(X_tr_num, X_va_num, X_te_num)
    X_tr_base = np.hstack([S_tr, C_tr])
    X_va_base = np.hstack([S_va, C_va])
    X_te_base = np.hstack([S_te, C_te])
    base_names = list(num_cols) + cat_names

    ols_base = LinearRegression().fit(X_tr_base, y_train)
    ols_rmse, ols_mae, ols_r2 = calc_metrics(y_val, ols_base.predict(X_va_base))
    print(f"Baseline (OLS):   val-RMSE={ols_rmse:.4f}, MAE={ols_mae:.4f}, R^2={ols_r2:.4f}")

    _, base_alpha, (base_rmse, base_mae, base_r2) = eval_linear_ridge(
        X_tr_base, y_train, X_va_base, y_val
    )
    print(
        f"Baseline (Ridge, alpha={base_alpha}): "
        f"val-RMSE={base_rmse:.4f}, MAE={base_mae:.4f}, R^2={base_r2:.4f}"
    )

    # Итоговая таблица экспериментов (трансформация -> val-RMSE)
    report = [
        {
            "Трансформация": "Baseline (Impute + OHE + Scale, Ridge)",
            "Признаков": X_tr_base.shape[1],
            "Alpha": base_alpha,
            "val-RMSE": base_rmse,
            "ΔRMSE к baseline": 0.0,
            "val-MAE": base_mae,
            "val-R2": base_r2,
            "Полезно": True,
        }
    ]

    def log_exp(title, X_tr, X_va, use_ridge=True):
        if use_ridge:
            _, a, (rmse, mae, r2) = eval_linear_ridge(X_tr, y_train, X_va, y_val)
        else:
            mdl = LinearRegression().fit(X_tr, y_train)
            rmse, mae, r2 = calc_metrics(y_val, mdl.predict(X_va))
            a = 0.0
        delta = rmse - base_rmse
        useful = rmse < base_rmse - 1e-6
        report.append(
            {
                "Трансформация": title,
                "Признаков": X_tr.shape[1],
                "Alpha": a,
                "val-RMSE": rmse,
                "ΔRMSE к baseline": delta,
                "val-MAE": mae,
                "val-R2": r2,
                "Полезно": useful,
            }
        )
        return rmse, a

    # ------------------------------------------------------------------
    # 3. Feature Engineering (оценка каждого преобразования по val-RMSE)
    # ------------------------------------------------------------------
    print("\n--- 3. Feature Engineering ---")

    # 3.1 Выбросы (IQR clipping по train)
    X_tr_iqr, X_va_iqr, X_te_iqr, n_clip_tr, n_clip_va = clip_iqr_splits(X_tr_num, X_va_num, X_te_num)
    print(f"IQR clipping: обрезано значений на train={n_clip_tr}, на val={n_clip_va}")
    S_tr_iqr, S_va_iqr, S_te_iqr = scale_train_val_test(X_tr_iqr, X_va_iqr, X_te_iqr)
    iqr_rmse, _ = log_exp("Выбросы (IQR clipping)", np.hstack([S_tr_iqr, C_tr]), np.hstack([S_va_iqr, C_va]))

    # Выбираем, работать ли дальше с IQR-обрезанными числами или исходными
    if iqr_rmse < base_rmse:
        cur_num_tr, cur_num_va, cur_num_te = X_tr_iqr, X_va_iqr, X_te_iqr
        cur_num_label = "IQR"
    else:
        cur_num_tr, cur_num_va, cur_num_te = X_tr_num, X_va_num, X_te_num
        cur_num_label = "Raw"

    # 3.2 Поэлементные преобразования: Log1p, Sqrt, Square, Box-Cox, Yeo-Johnson
    mono_candidates = {}

    # Log1p(x)
    tr_log, va_log, te_log = scale_train_val_test(
        np.log1p(cur_num_tr), np.log1p(cur_num_va), np.log1p(cur_num_te)
    )
    r_log, _ = log_exp("Log1p(x)", np.hstack([tr_log, C_tr]), np.hstack([va_log, C_va]))
    mono_candidates["Log1p(x)"] = (r_log, tr_log, va_log, te_log, [f"log1p_{c}" for c in num_cols])

    # Sqrt(x)
    tr_sqrt, va_sqrt, te_sqrt = scale_train_val_test(
        np.sqrt(cur_num_tr), np.sqrt(cur_num_va), np.sqrt(cur_num_te)
    )
    r_sqrt, _ = log_exp("Sqrt(x)", np.hstack([tr_sqrt, C_tr]), np.hstack([va_sqrt, C_va]))
    mono_candidates["Sqrt(x)"] = (r_sqrt, tr_sqrt, va_sqrt, te_sqrt, [f"sqrt_{c}" for c in num_cols])

    # Square(x) = x^2 (вторая возможная трактовка записи SQR(x))
    tr_sq, va_sq, te_sq = scale_train_val_test(
        cur_num_tr ** 2, cur_num_va ** 2, cur_num_te ** 2
    )
    r_sq, _ = log_exp("Square(x) = x^2", np.hstack([tr_sq, C_tr]), np.hstack([va_sq, C_va]))
    mono_candidates["Square(x)"] = (r_sq, tr_sq, va_sq, te_sq, [f"sq_{c}" for c in num_cols])

    # Box-Cox (строго положительные числа)
    pt_bc = PowerTransformer(method="box-cox", standardize=True)
    tr_bc = pt_bc.fit_transform(cur_num_tr)
    va_bc = pt_bc.transform(cur_num_va)
    te_bc = pt_bc.transform(cur_num_te)
    r_bc, _ = log_exp("Box-Cox", np.hstack([tr_bc, C_tr]), np.hstack([va_bc, C_va]))
    mono_candidates["Box-Cox"] = (r_bc, tr_bc, va_bc, te_bc, [f"bc_{c}" for c in num_cols])

    # Yeo-Johnson
    pt_yj = PowerTransformer(method="yeo-johnson", standardize=True)
    tr_yj = pt_yj.fit_transform(cur_num_tr)
    va_yj = pt_yj.transform(cur_num_va)
    te_yj = pt_yj.transform(cur_num_te)
    r_yj, _ = log_exp("Yeo-Johnson", np.hstack([tr_yj, C_tr]), np.hstack([va_yj, C_va]))
    mono_candidates["Yeo-Johnson"] = (r_yj, tr_yj, va_yj, te_yj, [f"yj_{c}" for c in num_cols])

    # 3.3 Полиномы degree=2: сначала без регуляризации (OLS), потом с регуляризацией (Ridge)
    S_cur_tr, S_cur_va, S_cur_te = scale_train_val_test(cur_num_tr, cur_num_va, cur_num_te)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    P_tr_raw = poly.fit_transform(S_cur_tr)
    P_va_raw = poly.transform(S_cur_va)
    P_te_raw = poly.transform(S_cur_te)
    P_tr, P_va, P_te = scale_train_val_test(P_tr_raw, P_va_raw, P_te_raw)
    poly_names = list(poly.get_feature_names_out(num_cols))

    log_exp(
        "PolynomialFeatures(deg=2) БЕЗ регуляризации (OLS)",
        np.hstack([P_tr, C_tr]),
        np.hstack([P_va, C_va]),
        use_ridge=False,
    )
    r_poly, _ = log_exp(
        "PolynomialFeatures(deg=2) + Ridge",
        np.hstack([P_tr, C_tr]),
        np.hstack([P_va, C_va]),
        use_ridge=True,
    )

    # 3.4 B-сплайны: SplineTransformer (3..10 узлов)
    best_spline_rmse = float("inf")
    best_spline_knots = 3
    best_spline_data = None
    for knots in [3, 5, 7, 10]:
        sp = SplineTransformer(n_knots=knots, degree=3, include_bias=False)
        sp_tr_raw = sp.fit_transform(S_cur_tr)
        sp_va_raw = sp.transform(S_cur_va)
        sp_te_raw = sp.transform(S_cur_te)
        sp_tr, sp_va, sp_te = scale_train_val_test(sp_tr_raw, sp_va_raw, sp_te_raw)
        sp_names = list(sp.get_feature_names_out(num_cols))
        r_sp, _ = log_exp(
            f"SplineTransformer (knots={knots}) + Ridge",
            np.hstack([sp_tr, C_tr]),
            np.hstack([sp_va, C_va]),
        )
        if r_sp < best_spline_rmse:
            best_spline_rmse = r_sp
            best_spline_knots = knots
            best_spline_data = (sp_tr, sp_va, sp_te, sp_names)

    # 3.5 Групповые агрегаты (mean/median/std по категориям)
    agg_cols = ["bmi", "bp", "s1", "s3", "s5"]
    agg_tr_raw, agg_va_raw, agg_te_raw, agg_names = make_group_aggregates(
        cur_num_tr, cur_num_va, cur_num_te, df_tr_cat, df_va_cat, df_te_cat, num_cols, agg_cols
    )
    agg_tr, agg_va, agg_te = scale_train_val_test(agg_tr_raw, agg_va_raw, agg_te_raw)
    r_agg, _ = log_exp(
        "Групповые агрегаты (mean/median/std) + базовые",
        np.hstack([S_cur_tr, agg_tr, C_tr]),
        np.hstack([S_cur_va, agg_va, C_va]),
    )

    # ------------------------------------------------------------------
    # Собираем лучшую комбинацию из ТОЛЬКО полезных преобразований
    # ------------------------------------------------------------------
    best_mono_name = min(mono_candidates, key=lambda k: mono_candidates[k][0])
    best_mono_rmse, m_tr, m_va, m_te, m_names = mono_candidates[best_mono_name]

    if best_mono_rmse < min(base_rmse, iqr_rmse):
        comb_tr, comb_va, comb_te = m_tr, m_va, m_te
        comb_names = list(m_names)
        chosen_parts = [f"{cur_num_label}+{best_mono_name}"]
    else:
        comb_tr, comb_va, comb_te = S_cur_tr, S_cur_va, S_cur_te
        comb_names = list(num_cols)
        chosen_parts = [cur_num_label]

    _, cur_alpha, (cur_best_rmse, _, _) = eval_linear_ridge(
        np.hstack([comb_tr, C_tr]), y_train, np.hstack([comb_va, C_va]), y_val
    )

    # Пробуем по очереди добавлять структурные блоки (оставляем только то, что снижает val-RMSE)
    # 1) Полиномы (или только квадратичные/парные члены поверх уже имеющихся)
    cand_tr = np.hstack([comb_tr, P_tr])
    cand_va = np.hstack([comb_va, P_va])
    _, _, (r_try, _, _) = eval_linear_ridge(
        np.hstack([cand_tr, C_tr]), y_train, np.hstack([cand_va, C_va]), y_val
    )
    if r_try < cur_best_rmse - 1e-6:
        comb_tr, comb_va, comb_te = cand_tr, cand_va, np.hstack([comb_te, P_te])
        comb_names.extend([f"poly_{n}" for n in poly_names])
        cur_best_rmse = r_try
        chosen_parts.append("Poly(deg=2)")

    # 2) Лучшие B-сплайны
    sp_tr, sp_va, sp_te, sp_names = best_spline_data
    cand_tr = np.hstack([comb_tr, sp_tr])
    cand_va = np.hstack([comb_va, sp_va])
    _, _, (r_try, _, _) = eval_linear_ridge(
        np.hstack([cand_tr, C_tr]), y_train, np.hstack([cand_va, C_va]), y_val
    )
    if r_try < cur_best_rmse - 1e-6:
        comb_tr, comb_va, comb_te = cand_tr, cand_va, np.hstack([comb_te, sp_te])
        comb_names.extend([f"sp_{n}" for n in sp_names])
        cur_best_rmse = r_try
        chosen_parts.append(f"Spline(knots={best_spline_knots})")

    # 3) Групповые агрегаты
    cand_tr = np.hstack([comb_tr, agg_tr])
    cand_va = np.hstack([comb_va, agg_va])
    _, _, (r_try, _, _) = eval_linear_ridge(
        np.hstack([cand_tr, C_tr]), y_train, np.hstack([cand_va, C_va]), y_val
    )
    if r_try < cur_best_rmse - 1e-6:
        comb_tr, comb_va, comb_te = cand_tr, cand_va, np.hstack([comb_te, agg_te])
        comb_names.extend(agg_names)
        cur_best_rmse = r_try
        chosen_parts.append("GroupAgg")

    X_tr_comb = np.hstack([comb_tr, C_tr])
    X_va_comb = np.hstack([comb_va, C_va])
    X_te_comb = np.hstack([comb_te, C_te])
    comb_names = comb_names + cat_names

    # Убираем возможные вырожденные/нулевые столбцы (std == 0 на train)
    stds = X_tr_comb.std(axis=0)
    keep_nonzero = stds > 1e-12
    X_tr_comb = X_tr_comb[:, keep_nonzero]
    X_va_comb = X_va_comb[:, keep_nonzero]
    X_te_comb = X_te_comb[:, keep_nonzero]
    comb_names = [n for n, k in zip(comb_names, keep_nonzero) if k]

    # Финальная стандартизация собранной матрицы (для стабильности SGD и регуляризации)
    X_tr_comb, X_va_comb, X_te_comb = scale_train_val_test(X_tr_comb, X_va_comb, X_te_comb)
    comb_rmse, comb_alpha = log_exp(
        f"Комбинация полезных ({' + '.join(chosen_parts)})",
        X_tr_comb,
        X_va_comb,
    )

    # ------------------------------------------------------------------
    # 3.6 Отбор признаков (корреляция, mutual information, RFE, Lasso)
    # ------------------------------------------------------------------
    print("\n--- Отбор признаков (обучение только на train) ---")
    p_total = X_tr_comb.shape[1]
    selection_variants = {}

    # 1) Корреляция с y_train
    abs_corr = np.abs(
        [np.corrcoef(X_tr_comb[:, j], y_train)[0, 1] for j in range(p_total)]
    )
    best_corr_mask = np.ones(p_total, dtype=bool)
    best_corr_rmse = float("inf")
    for thr in [0.03, 0.05, 0.08, 0.12, 0.15, 0.20]:
        mask = abs_corr >= thr
        if mask.sum() < 2:
            continue
        _, _, (r_c, _, _) = eval_linear_ridge(X_tr_comb[:, mask], y_train, X_va_comb[:, mask], y_val)
        if r_c < best_corr_rmse:
            best_corr_rmse = r_c
            best_corr_mask = mask
    log_exp("Отбор: корреляция с target", X_tr_comb[:, best_corr_mask], X_va_comb[:, best_corr_mask])
    selection_variants["Корреляция"] = (best_corr_rmse, best_corr_mask)

    # 2) Mutual Information
    mi_scores = mutual_info_regression(X_tr_comb, y_train, random_state=SEED)
    best_mi_mask = np.ones(p_total, dtype=bool)
    best_mi_rmse = float("inf")
    k_grid = sorted(set([max(2, p_total // 4), max(3, p_total // 2), max(4, (3 * p_total) // 4), p_total]))
    for k in k_grid:
        top_idx = np.argsort(mi_scores)[-k:]
        mask = np.zeros(p_total, dtype=bool)
        mask[top_idx] = True
        _, _, (r_m, _, _) = eval_linear_ridge(X_tr_comb[:, mask], y_train, X_va_comb[:, mask], y_val)
        if r_m < best_mi_rmse:
            best_mi_rmse = r_m
            best_mi_mask = mask
    log_exp("Отбор: Mutual Information", X_tr_comb[:, best_mi_mask], X_va_comb[:, best_mi_mask])
    selection_variants["MutualInfo"] = (best_mi_rmse, best_mi_mask)

    # 3) RFE с Ridge
    best_rfe_mask = np.ones(p_total, dtype=bool)
    best_rfe_rmse = float("inf")
    for k in k_grid:
        if k == p_total:
            mask = np.ones(p_total, dtype=bool)
        else:
            rfe = RFE(estimator=Ridge(alpha=comb_alpha, random_state=SEED), n_features_to_select=k)
            rfe.fit(X_tr_comb, y_train)
            mask = rfe.support_
        _, _, (r_r, _, _) = eval_linear_ridge(X_tr_comb[:, mask], y_train, X_va_comb[:, mask], y_val)
        if r_r < best_rfe_rmse:
            best_rfe_rmse = r_r
            best_rfe_mask = mask
    log_exp("Отбор: RFE (Ridge)", X_tr_comb[:, best_rfe_mask], X_va_comb[:, best_rfe_mask])
    selection_variants["RFE"] = (best_rfe_rmse, best_rfe_mask)

    # 4) LassoCV
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=20000).fit(X_tr_comb, y_train)
    lasso_mask = np.abs(lasso.coef_) > 1e-6
    if lasso_mask.sum() < 2:
        lasso_mask = np.ones(p_total, dtype=bool)
    r_lasso, _ = log_exp("Отбор: LassoCV", X_tr_comb[:, lasso_mask], X_va_comb[:, lasso_mask])
    selection_variants["LassoCV"] = (r_lasso, lasso_mask)

    # Выбираем итоговый набор признаков: берём отбор, если он не ухудшает val-RMSE
    best_sel_name = min(selection_variants, key=lambda k: selection_variants[k][0])
    best_sel_rmse, best_sel_mask = selection_variants[best_sel_name]

    if best_sel_rmse <= comb_rmse:
        final_mask = best_sel_mask
        print(f"Применён отбор «{best_sel_name}»: оставлено {final_mask.sum()} из {p_total} признаков.")
    else:
        final_mask = np.ones(p_total, dtype=bool)
        print(f"Отбор не улучшил комбинацию, оставляем все {p_total} отобранных признаков.")

    X_train_final = X_tr_comb[:, final_mask]
    X_val_final = X_va_comb[:, final_mask]
    X_test_final = X_te_comb[:, final_mask]
    final_names = [n for n, m in zip(comb_names, final_mask) if m]

    _, best_alpha_final, (final_val_rmse, final_val_mae, final_val_r2) = eval_linear_ridge(
        X_train_final, y_train, X_val_final, y_val
    )
    assert final_val_rmse <= base_rmse + 1e-6, "Итоговый val-RMSE должен быть <= baseline!"

    report_df = pd.DataFrame(report)
    report_path = os.path.join(OUT_DIR, "transformation_report.csv")
    report_df.to_csv(report_path, index=False, float_format="%.4f")

    print("\n--- Таблица «трансформация -> val-RMSE» ---")
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # График сравнения трансформаций по val-RMSE
    plot_rep = report_df.iloc[::-1].reset_index(drop=True)
    plt.figure(figsize=(10.5, 6.8))
    bar_colors = ["seagreen" if u else "lightcoral" for u in plot_rep["Полезно"]]
    plt.barh(plot_rep["Трансформация"], plot_rep["val-RMSE"], color=bar_colors)
    plt.axvline(base_rmse, color="navy", linestyle="--", lw=1.3, label=f"Baseline = {base_rmse:.2f}")
    plt.xlabel("val-RMSE (меньше — лучше)")
    plt.title("Влияние Feature Engineering и отбора признаков на val-RMSE")
    plt.legend(loc="lower right")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fe_comparison.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 4. Четыре реализации МНК (Normal Equations, SVD, PyTorch SGD, sklearn)
    # ------------------------------------------------------------------
    print("\n--- 4. Четыре реализации МНК (своими руками + sklearn) ---")
    cond_num = np.linalg.cond(X_train_final)
    print(
        f"Итоговая матрица: {X_train_final.shape}, число обусловленности cond(X) = {cond_num:.2f}, "
        f"лучшее alpha (Ridge) = {best_alpha_final}"
    )

    ols_rows = []
    # Сначала чистый МНК (alpha = 0.0), затем (если alpha > 0) регуляризованная модель (alpha = best_alpha_final)
    alpha_modes = [(0.0, "Чистый МНК (alpha=0)")]
    if best_alpha_final > 0.0:
        alpha_modes.append((best_alpha_final, f"МНК + L2 (alpha={best_alpha_final})"))

    sgd_curves = None
    last_test_preds = None

    for alpha_val, mode_label in alpha_modes:
        print(f"\nРежим: {mode_label}")
        t0 = time.perf_counter()
        if alpha_val == 0.0:
            sk_model = LinearRegression().fit(X_train_final, y_train)
        else:
            sk_model = Ridge(alpha=alpha_val, random_state=SEED).fit(X_train_final, y_train)
        t_sk = (time.perf_counter() - t0) * 1000.0
        w_sk = sk_model.coef_
        b_sk = float(sk_model.intercept_)

        t0 = time.perf_counter()
        w_ne, b_ne = fit_normal_equations(X_train_final, y_train, alpha=alpha_val)
        t_ne = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        w_svd, b_svd = fit_svd(X_train_final, y_train, alpha=alpha_val)
        t_svd = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        w_sgd, b_sgd, tr_hist, va_hist = fit_sgd_pytorch(
            X_train_final,
            y_train,
            X_val_final,
            y_val,
            alpha=alpha_val,
            lr=0.02,
            epochs=2000,
            batch_size=64,
        )
        t_sgd = (time.perf_counter() - t0) * 1000.0

        for method_name, w_m, b_m, ms in [
            ("Normal Equations", w_ne, b_ne, t_ne),
            ("SVD", w_svd, b_svd, t_svd),
            ("SGD (PyTorch)", w_sgd, b_sgd, t_sgd),
            ("sklearn", w_sk, b_sk, t_sk),
        ]:
            tr_rmse, _, _ = calc_metrics(y_train, X_train_final @ w_m + b_m)
            va_rmse, _, _ = calc_metrics(y_val, X_val_final @ w_m + b_m)
            te_pred = X_test_final @ w_m + b_m
            te_rmse, te_mae, te_r2 = calc_metrics(y_test, te_pred)
            max_w_diff = float(np.max(np.abs(w_m - w_sk)))

            ols_rows.append(
                {
                    "Режим": mode_label,
                    "Метод": method_name,
                    "Train RMSE": tr_rmse,
                    "Val RMSE": va_rmse,
                    "Test RMSE": te_rmse,
                    "Test MAE": te_mae,
                    "Test R2": te_r2,
                    "Max |w - w_sklearn|": max_w_diff,
                    "Время (мс)": ms,
                }
            )

        # Сохраняем кривые и предсказания для итогового графика
        sgd_curves = (tr_hist, va_hist, calc_metrics(y_val, X_val_final @ w_svd + b_svd)[0], mode_label)
        last_test_preds = (X_test_final @ w_svd + b_svd, X_test_final @ w_sgd + b_sgd, mode_label)

    ols_df = pd.DataFrame(ols_rows)
    ols_path = os.path.join(OUT_DIR, "ols_comparison.csv")
    ols_df.to_csv(ols_path, index=False, float_format="%.6f")
    print("\n--- Сравнение 4 реализаций МНК ---")
    print(ols_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # График сходимости PyTorch SGD и scatter на test
    tr_hist, va_hist, svd_val_rmse, mode_label = sgd_curves
    pred_svd_test, pred_sgd_test, _ = last_test_preds

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ep = np.arange(1, len(tr_hist) + 1)
    axes[0].plot(ep, tr_hist, label="SGD Train RMSE", lw=1.8)
    axes[0].plot(ep, va_hist, label="SGD Val RMSE", lw=1.8)
    axes[0].axhline(svd_val_rmse, color="black", linestyle="--", lw=1.2, label=f"Точный SVD Val ({svd_val_rmse:.2f})")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"Сходимость SGD (PyTorch): {mode_label}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_test, pred_svd_test, alpha=0.75, label="SVD / sklearn", s=28)
    axes[1].scatter(y_test, pred_sgd_test, alpha=0.5, marker="x", label="PyTorch SGD", s=32)
    lim_min = min(y_test.min(), pred_svd_test.min()) - 5
    lim_max = max(y_test.max(), pred_svd_test.max()) + 5
    axes[1].plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="Идеал y = y_hat")
    axes[1].set_xlabel("Истинный target (test)")
    axes[1].set_ylabel("Предсказание")
    axes[1].set_title("Качество предсказания на test")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ols_sgd_and_test.png"), dpi=150)
    plt.close()

    print(f"\nИтоговые признаки ({len(final_names)}): {final_names[:10]}{'...' if len(final_names) > 10 else ''}")
    print(f"Результаты и графики сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()
