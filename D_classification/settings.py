# analysis fileを指定
path = r"C:\Users\rzxcv\Downloads\Zurich_Red\analysis_file\251105_152737"

# サンプルごとに使用する範囲を指定 (例: {クラス: (開始インデックス, 終了インデックス)})
# クラスは各サンプルのinfo.jsonに記載されている順番に対応する
# 番号を指定せず、すべてのピークを学習に使う場合は「selected_peak_idx = None」とする
selected_peak_idx = None
# ハイパラ最適化に使うパラメータ
trial_num = None # {int or None}
study_num = None # {int or None}

# cnnに使うパラメータ
batch_size = None # {int or None}
learning_rate = 0.0001 # {float or None}
weight_decay = None # {float or None}
dropout_rate = None # {flort or None}

num_layers = None # {int or None}
num_filters = None # {int or None}
out_features = None # {int or None}

max_epoch = None # {int or None}

seed = None