import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import settings

# --- データ読み込み ---
X = np.load(os.path.join(settings.path, 'X.npy'), allow_pickle=True)
y = np.load(os.path.join(settings.path, 'y.npy'), allow_pickle=True)
n_samples, data_len, feat_num = X.shape

# --- 出力ディレクトリ ---
output_dir = os.path.join(settings.path, 'dbscan_clustering')
os.makedirs(output_dir, exist_ok=True)

# --- 特徴ベクトル化 ---
X_flat = X.reshape(n_samples, -1)
use_pca = getattr(settings, 'use_pca', True)
pca_components = getattr(settings, 'pca_components', 50)
if use_pca:
    pca = PCA(n_components=pca_components, random_state=getattr(settings, 'seed', 42))
    X_feat = pca.fit_transform(X_flat)
else:
    X_feat = X_flat

# --- DBSCANクラスタリング ---
# eps, min_samplesはデータに応じて調整してください
# epsが小さすぎると全てノイズになることもあるので注意
# 例: eps=2, min_samples=10

dbscan = DBSCAN(eps=0.002, min_samples=30)
labels = dbscan.fit_predict(X_feat)

# --- クラスタ数・ノイズ数の出力 ---
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
num_noise = np.sum(labels == -1)
print(f'クラスタ数: {num_clusters}, ノイズ数: {num_noise}')

# --- シルエットスコア（ノイズ除外）---
if num_clusters > 1:
    score = silhouette_score(X_feat[labels != -1], labels[labels != -1])
    print(f'Silhouette score (ノイズ除外): {score:.3f}')
else:
    print('クラスタが1つ以下のためスコア計算不可')

# --- クラスタごとの分布をテーブル出力 ---
df = pd.DataFrame({'cluster': labels, 'class': y})
ct = pd.crosstab(df['cluster'], df['class'], normalize='index')
ct.to_csv(os.path.join(output_dir, 'cluster_class_distribution.csv'))

# --- ノイズデータのインデックス ---
noise_idx = np.where(labels == -1)[0]
valid_idx = np.where(labels != -1)[0]

# --- 有効データのみ保存 ---
X_valid = X[valid_idx]
y_valid = y[valid_idx]
np.save(os.path.join(output_dir, 'X_valid.npy'), X_valid)
np.save(os.path.join(output_dir, 'y_valid.npy'), y_valid)
print(f'有効データ数: {len(valid_idx)}')

# --- ノイズデータも保存（必要なら） ---
X_noise = X[noise_idx]
y_noise = y[noise_idx]
np.save(os.path.join(output_dir, 'X_noise.npy'), X_noise)
np.save(os.path.join(output_dir, 'y_noise.npy'), y_noise)

# --- 各クラスタの代表波形を可視化 ---
for c in set(labels):
    if c == -1:
        continue
    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        continue
    X_cluster = X[idx]
    plt.figure(figsize=(8, 4))
    for j in range(min(10, len(X_cluster))):
        for ch in range(feat_num):
            plt.plot(X_cluster[j, :, ch], alpha=0.3, label=f'ch{ch}' if j == 0 else "")
    plt.title(f'Cluster {c} (例:最大10波形)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'cluster_{c}_examples.png'))
    plt.close()

# --- ノイズ波形も可視化 ---
if len(noise_idx) > 0:
    plt.figure(figsize=(8, 4))
    for j in range(min(10, len(X_noise))):
        for ch in range(feat_num):
            plt.plot(X_noise[j, :, ch], alpha=0.3, label=f'ch{ch}' if j == 0 else "")
    plt.title('Noise cluster (例:最大10波形)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'noise_examples.png'))
    plt.close()

# --- アドバイス: 精度向上のために ---
# 1. PCAやt-SNE, UMAPなどで次元削減し、特徴量の分布を可視化・調整
# 2. DBSCANのeps, min_samplesはデータごとに最適化（グリッドサーチや可視化で調整）
# 3. 標準化・正規化（StandardScaler, MinMaxScaler）を事前に適用
# 4. 波形の前処理（ノイズ除去, スムージング, 正規化）
# 5. 特徴量設計（ピーク値, RMS, 周波数成分など）
# 6. クラスタごとの代表波形や平均波形を比較し、異常検知の閾値を検討
# 7. ノイズクラスタの波形を目視確認し、閾値や前処理を再検討
# 8. 必要に応じて他のクラスタリング手法（HDBSCAN, GaussianMixture等）も検討

print('DBSCANクラスタリング完了')
