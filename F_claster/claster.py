import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import settings

# データ読み込み
X = np.load(os.path.join(settings.path, 'X.npy'), allow_pickle=True)
y = np.load(os.path.join(settings.path, 'y.npy'), allow_pickle=True)
n_samples, data_len, feat_num = X.shape

# 出力ディレクトリ
output_dir = os.path.join(settings.path, 'clustering')
os.makedirs(output_dir, exist_ok=True)

# 特徴ベクトル化
X_flat = X.reshape(n_samples, -1)
if settings.use_pca:
    pca = PCA(n_components=settings.pca_components, random_state=settings.seed)
    X_feat = pca.fit_transform(X_flat)
else:
    X_feat = X_flat

# KMeans クラスタリング
kmeans = KMeans(n_clusters=settings.cluster_num, random_state=settings.seed)
labels = kmeans.fit_predict(X_feat)

# シルエットスコア
score = silhouette_score(X_feat, labels)
print(f'Silhouette score: {score:.3f}')

# クラスごとの分布をテーブル出力
df = pd.DataFrame({'cluster': labels, 'class': y})
ct = pd.crosstab(df['cluster'], df['class'], normalize='index')
ct.to_csv(os.path.join(output_dir, 'cluster_class_distribution.csv'))

# 各クラスタ中心波形のプロット
centers = kmeans.cluster_centers_
if settings.use_pca:
    centers = pca.inverse_transform(centers)
centers_ts = centers.reshape(settings.cluster_num, data_len, feat_num)

for i, c in enumerate(centers_ts):
    plt.figure()
    for ch in range(feat_num):
        plt.plot(c[:, ch], label=f'ch{ch}')
    plt.title(f'Cluster {i}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'cluster_{i}.png'))
    plt.close()

# --- クラスタごとにX, yを分割して保存 ---
for i in range(settings.cluster_num):
    idx = np.where(labels == i)[0]
    X_cluster = X[idx]
    y_cluster = y[idx]
    np.save(os.path.join(output_dir, f'cluster_{i}_X.npy'), X_cluster)
    np.save(os.path.join(output_dir, f'cluster_{i}_y.npy'), y_cluster)
    print(f'Cluster {i}: X shape =', X_cluster.shape, ', y shape =', y_cluster.shape)

# --- クラスタごとに代表的な波形を可視化 ---
num_examples = 5  # 各クラスタで可視化する波形数
for i in range(settings.cluster_num):
    idx = np.where(labels == i)[0]
    if len(idx) == 0:
        continue
    X_cluster = X[idx]
    # 特徴空間で中心に近い順に並べる
    if settings.use_pca:
        X_cluster_feat = pca.transform(X_cluster.reshape(len(idx), -1))
    else:
        X_cluster_feat = X_cluster.reshape(len(idx), -1)
    dists = cdist(X_cluster_feat, [kmeans.cluster_centers_[i]])[:, 0]
    nearest_idx = np.argsort(dists)[:num_examples]
    for n, j in enumerate(nearest_idx):
        plt.figure()
        for ch in range(feat_num):
            plt.plot(X_cluster[j, :, ch], label=f'ch{ch}')
        plt.title(f'Cluster {i} Example {n+1}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'cluster_{i}_example_{n+1}.png'))
        plt.close()

# --- 各クラスタ内の全波形を1枚に重ねてプロット ---
for i in range(settings.cluster_num):
    idx = np.where(labels == i)[0]
    if len(idx) == 0:
        continue
    X_cluster = X[idx]
    plt.figure(figsize=(8, 4))
    for j in range(len(X_cluster)):
        for ch in range(feat_num):
            plt.plot(X_cluster[j, :, ch], alpha=0.2, label=f'ch{ch}' if j == 0 else "")
    plt.title(f'All waveforms in Cluster {i}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'cluster_{i}_all_waveforms.png'))
    plt.close()

# --- 全データをクラスタごとに色分けして重ねてプロット ---
colors = plt.cm.get_cmap('tab10', settings.cluster_num)
plt.figure(figsize=(10, 5))
for i in range(settings.cluster_num):
    idx = np.where(labels == i)[0]
    if len(idx) == 0:
        continue
    for j in idx:
        for ch in range(feat_num):
            plt.plot(X[j, :, ch], color=colors(i), alpha=0.15)
plt.title('All waveforms colored by cluster')
plt.savefig(os.path.join(output_dir, 'all_waveforms_by_cluster.png'))
plt.close()