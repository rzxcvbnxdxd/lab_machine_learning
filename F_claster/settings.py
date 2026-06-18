# analysis fileを指定
path = r"C:\Users\rzxcv\Downloads\analysis_file\analysis_file\250514_192117"

# サンプルごとに使用する範囲を指定 (例: {クラス: (開始インデックス, 終了インデックス)})
# クラスは各サンプルのinfo.jsonに記載されている順番に対応する
# 番号を指定せず、すべてのピークを学習に使う場合は「selected_peak_idx = None」とする
selected_peak_idx = None
selected_peak_idx = {
    
#     0: (0, 600),  # クラス0の0番目から50番目
#     1: (100, 700),  # クラス1の10番目から60番目
#     2: (200, 800),  # クラス2の20番目から70番目
    # 他のクラスも必要に応じて指定
}

# クラスタリング設定
cluster_num    = 5        # クラスタ数
use_pca        = True     # 前処理として PCA を使う
pca_components = 50       # PCA 次元数
seed           = 42       # 乱数シード
