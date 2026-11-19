"""
Лабораторная работа №3: кластеризация и понижение размерности (50+ признаков).
1) Датасет с 50+ признаками (sklearn load_digits: 1797 x 64)
2) Кластеризация ДО (K-means и GMM) + сохранение результатов
3) Понижение размерности: PCA, t-SNE и автоэнкодер (PyTorch)
4) Кластеризация ПОСЛЕ (K-means и GMM) на сжатых представлениях
5) Сравнение результатов ДО / ПОСЛЕ
"""

import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# Фиксируем все генераторы случайных чисел
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "lab3")
os.makedirs(OUT_DIR, exist_ok=True)


class Autoencoder(nn.Module):
    """Простой полносвязный автоэнкодер на PyTorch: 64 -> 32 -> 16 -> 10 -> 16 -> 32 -> 64."""

    def __init__(self, input_dim=64, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z


def eval_clustering(X_repr, labels_pred, y_true):
    """Считает внутренние и внешние метрики качества кластеризации."""
    sil = silhouette_score(X_repr, labels_pred)
    db = davies_bouldin_score(X_repr, labels_pred)
    ch = calinski_harabasz_score(X_repr, labels_pred)
    ari = adjusted_rand_score(y_true, labels_pred)
    nmi = normalized_mutual_info_score(y_true, labels_pred)
    return {
        "Silhouette": sil,
        "Davies-Bouldin": db,
        "Calinski-Harabasz": ch,
        "ARI": ari,
        "NMI": nmi,
    }


def run_kmeans_and_gmm(X_repr, y_true, repr_name, dim, n_clusters=10):
    """Запускает K-means и GMM на данном представлении и возвращает метки и метрики."""
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
    labels_km = km.fit_predict(X_repr)
    m_km = eval_clustering(X_repr, labels_km, y_true)
    m_km["Представление"] = repr_name
    m_km["Размерность"] = dim
    m_km["Алгоритм"] = "K-means"

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        reg_covar=1e-4,
        n_init=5,
        random_state=SEED,
    )
    labels_gmm = gmm.fit_predict(X_repr)
    m_gmm = eval_clustering(X_repr, labels_gmm, y_true)
    m_gmm["Представление"] = repr_name
    m_gmm["Размерность"] = dim
    m_gmm["Алгоритм"] = "GMM"

    return labels_km, labels_gmm, [m_km, m_gmm]


def main():
    print("=" * 74)
    print("ЛАБОРАТОРНАЯ РАБОТА №3: КЛАСТЕРИЗАЦИЯ И ПОНИЖЕНИЕ РАЗМЕРНОСТИ")
    print("=" * 74)

    # ------------------------------------------------------------------
    # 1. Датасет с 50+ признаками (sklearn load_digits)
    # ------------------------------------------------------------------
    digits = load_digits()
    X_raw = digits.data
    y_true = digits.target
    assert X_raw.shape[1] >= 50, "Требуется датасет с 50+ признаками!"

    print("\n--- 1. Датасет и масштабирование ---")
    print(f"Датасет: sklearn load_digits (оффлайн)")
    print(f"Объектов: {X_raw.shape[0]}, признаков: {X_raw.shape[1]} (>= 50), классов: {len(np.unique(y_true))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Проверка выбора числа кластеров (метод локтя и силуэта)
    print("\nПроверка числа кластеров k (метод локтя и Silhouette на исходных 64 признаках):")
    for k in [6, 8, 10, 12, 14]:
        km_tmp = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
        sil_tmp = silhouette_score(X_scaled, km_tmp.labels_)
        print(f"  k = {k:2d} | Inertia = {km_tmp.inertia_:9.1f} | Silhouette = {sil_tmp:.4f}")

    n_clusters = 10
    clusters_df = pd.DataFrame({"true_digit": y_true})
    all_metrics = []

    # ------------------------------------------------------------------
    # 2. Кластеризация ДО понижения размерности (64 признака)
    # ------------------------------------------------------------------
    print("\n--- 2. Кластеризация ДО понижения размерности (Raw 64D) ---")
    lbl_raw_km, lbl_raw_gmm, m_raw = run_kmeans_and_gmm(
        X_scaled, y_true, "Raw (64D)", 64, n_clusters=n_clusters
    )
    clusters_df["raw64_kmeans"] = lbl_raw_km
    clusters_df["raw64_gmm"] = lbl_raw_gmm
    all_metrics.extend(m_raw)

    for row in m_raw:
        print(
            f"  {row['Алгоритм']:8s} | Sil={row['Silhouette']:.4f} | DB={row['Davies-Bouldin']:.4f} | "
            f"CH={row['Calinski-Harabasz']:7.2f} | ARI={row['ARI']:.4f} | NMI={row['NMI']:.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Понижение размерности: PCA, t-SNE и автоэнкодер (PyTorch)
    # ------------------------------------------------------------------
    print("\n--- 3. Понижение размерности: PCA, t-SNE и Autoencoder (PyTorch) ---")

    # 3a) PCA
    pca_full = PCA(random_state=SEED).fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    print(f"PCA: 10 компонент объясняют {cum_var[9]:.2%} дисперсии, 90% дисперсии -> {n_90} компонент")

    Z_pca10 = PCA(n_components=10, random_state=SEED).fit_transform(X_scaled)
    Z_pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)

    # 3b) t-SNE
    print("Запуск t-SNE (2D, perplexity=30)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    )
    Z_tsne2 = tsne.fit_transform(X_scaled)

    # 3c) Автоэнкодер на PyTorch (64 -> 32 -> 16 -> 10)
    print("Обучение автоэнкодера PyTorch (64 -> 32 -> 16 -> 10)...")
    device = torch.device("cpu")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dl_gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=dl_gen)

    ae_model = Autoencoder(input_dim=X_scaled.shape[1], latent_dim=10).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

    ae_losses = []
    epochs = 120
    for epoch in range(1, epochs + 1):
        ae_model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, _ = ae_model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(dataset)
        ae_losses.append(epoch_loss)
        if epoch == 1 or epoch % 20 == 0:
            print(f"  Эпоха {epoch:3d}/{epochs} | Reconstruction MSE = {epoch_loss:.4f}")

    ae_model.eval()
    with torch.no_grad():
        x_recon, z_tensor = ae_model(X_tensor)
        final_recon_mse = criterion(x_recon, X_tensor).item()
        Z_ae10 = z_tensor.cpu().numpy()
    print(f"Итоговая ошибка реконструкции AE (MSE): {final_recon_mse:.4f}")

    # ------------------------------------------------------------------
    # 4. Кластеризация ПОСЛЕ понижения размерности
    # ------------------------------------------------------------------
    print("\n--- 4. Кластеризация ПОСЛЕ понижения размерности ---")
    representations = [
        ("PCA (10D)", 10, Z_pca10, "pca10"),
        ("PCA (2D)", 2, Z_pca2, "pca2"),
        ("t-SNE (2D)", 2, Z_tsne2, "tsne2"),
        ("Autoencoder (10D)", 10, Z_ae10, "ae10"),
    ]

    for repr_name, dim, Z_repr, col_prefix in representations:
        lbl_km, lbl_gmm, m_list = run_kmeans_and_gmm(
            Z_repr, y_true, repr_name, dim, n_clusters=n_clusters
        )
        clusters_df[f"{col_prefix}_kmeans"] = lbl_km
        clusters_df[f"{col_prefix}_gmm"] = lbl_gmm
        all_metrics.extend(m_list)

    clusters_path = os.path.join(OUT_DIR, "clusters_before_after.csv")
    clusters_df.to_csv(clusters_path, index=False)

    # ------------------------------------------------------------------
    # 5. Сравнение результатов ДО / ПОСЛЕ
    # ------------------------------------------------------------------
    comp_df = pd.DataFrame(all_metrics)

    # Считаем прирост ΔSilhouette и ΔARI относительно Raw (64D) для каждого алгоритма
    base_lookup = (
        comp_df[comp_df["Представление"] == "Raw (64D)"]
        .set_index("Алгоритм")[["Silhouette", "ARI"]]
        .to_dict("index")
    )
    comp_df["ΔSilhouette"] = comp_df.apply(
        lambda r: r["Silhouette"] - base_lookup[r["Алгоритм"]]["Silhouette"], axis=1
    )
    comp_df["ΔARI"] = comp_df.apply(
        lambda r: r["ARI"] - base_lookup[r["Алгоритм"]]["ARI"], axis=1
    )

    cols_order = [
        "Представление",
        "Размерность",
        "Алгоритм",
        "Silhouette",
        "Davies-Bouldin",
        "Calinski-Harabasz",
        "ARI",
        "NMI",
        "ΔSilhouette",
        "ΔARI",
    ]
    comp_df = comp_df[cols_order]
    comp_path = os.path.join(OUT_DIR, "comparison_before_after.csv")
    comp_df.to_csv(comp_path, index=False, float_format="%.4f")

    print("\n--- 5. Сравнение ДО / ПОСЛЕ понижения размерности ---")
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Краткие выводы по результатам
    best_ari_row = comp_df.loc[comp_df["ARI"].idxmax()]
    best_sil_row = comp_df.loc[comp_df["Silhouette"].idxmax()]
    print("\nВыводы:")
    print(
        f"- Лучший ARI (соответствие истинным цифрам): {best_ari_row['Алгоритм']} на "
        f"{best_ari_row['Представление']} (ARI = {best_ari_row['ARI']:.4f}, ΔARI = {best_ari_row['ΔARI']:+.4f})."
    )
    print(
        f"- Лучший Silhouette (компактность кластеров): {best_sil_row['Алгоритм']} на "
        f"{best_sil_row['Представление']} (Silhouette = {best_sil_row['Silhouette']:.4f}, "
        f"ΔSil = {best_sil_row['ΔSilhouette']:+.4f})."
    )

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------
    # 1) Накопленная дисперсия PCA + кривая обучения автоэнкодера
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, len(cum_var) + 1), cum_var, marker="o", ms=3)
    axes[0].axhline(0.90, color="r", linestyle="--", label="90% дисперсии")
    axes[0].axvline(10, color="g", linestyle=":", label="10 компонент")
    axes[0].set_xlabel("Число компонент PCA")
    axes[0].set_ylabel("Накопленная объяснённая дисперсия")
    axes[0].set_title("PCA: объяснённая дисперсия")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(1, epochs + 1), ae_losses, lw=2, color="tab:purple")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("MSE реконструкции")
    axes[1].set_title("Автоэнкодер PyTorch: кривая обучения")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_and_ae_loss.png"), dpi=150)
    plt.close()

    # 2) Визуализация 2D-проекций: истинные классы (верх) vs K-means кластеры (низ)
    Z_ae_vis2 = PCA(n_components=2, random_state=SEED).fit_transform(Z_ae10)
    panels = [
        ("PCA 2D", Z_pca2, clusters_df["pca2_kmeans"].values),
        ("t-SNE 2D", Z_tsne2, clusters_df["tsne2_kmeans"].values),
        ("AE 10D -> PCA 2D", Z_ae_vis2, clusters_df["ae10_kmeans"].values),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col_idx, (title, Z2, km_lbls) in enumerate(panels):
        axes[0, col_idx].scatter(
            Z2[:, 0], Z2[:, 1], c=y_true, cmap="tab10", s=12, alpha=0.8
        )
        axes[0, col_idx].set_title(f"{title}: истинные цифры")
        axes[0, col_idx].grid(alpha=0.2)

        axes[1, col_idx].scatter(
            Z2[:, 0], Z2[:, 1], c=km_lbls, cmap="tab10", s=12, alpha=0.8
        )
        axes[1, col_idx].set_title(f"{title}: кластеры K-means")
        axes[1, col_idx].grid(alpha=0.2)

    plt.suptitle("Проекции данных и кластеризация K-means", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "embeddings_2d.png"), dpi=150)
    plt.close()

    # 3) Сравнение метрик ДО и ПОСЛЕ (K-means и GMM)
    repr_order = ["Raw (64D)", "PCA (10D)", "PCA (2D)", "t-SNE (2D)", "Autoencoder (10D)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for ax, algo in zip(axes, ["K-means", "GMM"]):
        sub = comp_df[comp_df["Алгоритм"] == algo].set_index("Представление").loc[repr_order]
        x = np.arange(len(repr_order))
        w = 0.26
        ax.bar(x - w, sub["Silhouette"], width=w, label="Silhouette")
        ax.bar(x, sub["ARI"], width=w, label="ARI")
        ax.bar(x + w, sub["NMI"], width=w, label="NMI")
        ax.set_xticks(x)
        ax.set_xticklabels(repr_order, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{algo}: ДО vs ПОСЛЕ понижения размерности")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metrics_before_after.png"), dpi=150)
    plt.close()

    print(f"\nРезультаты и графики сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()
