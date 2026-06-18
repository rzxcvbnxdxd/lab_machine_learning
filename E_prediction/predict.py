import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os, sys, json, shutil

import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn

import function

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- 適宜変更 -------------

# 学習済みモデル(model.pth)が入っているディレクトリまでの絶対パスを指定
model_dir = r"C:\Users\rzxcv\Downloads\Zurich_Red\analysis_file\251105_152737\classification_file\cnn\251105_163329"
# --------------------------------------

# 分類モデルの情報を取得
shutil.copyfile(rf"{model_dir}\network.py", r"C:\Users\rzxcv\OneDrive\Desktop\研究\kenkyu\E_prediction\network.py")
import network

with open(f'{model_dir}/info.json', 'r', encoding='utf-8') as f:
    info_json = json.load(f)
measurement_freqs = '_'.join(info_json["dataset_info"]["measurement_freqs"])
baseline_correct_flag = info_json["dataset_info"]["baseline_correct_flag"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# 測定データの選択
# root = tk.Tk()
# root.attributes('-topmost', True) # エクスプローラーを最前面に表示
# root.withdraw()

# data_path = filedialog.askopenfilename(title="select measurement data", filetypes=[('file', '*.txt *.csv *.npy')])
data_path=r"C:\Users\rzxcv\Downloads\Zurich_Red\analysis_file\251105_152737\X.npy"
if not data_path:
    sys.exit()

else:
    print(f'selected file: {data_path}')

data_dir, data_file = os.path.split(data_path)

def evaluation(net, data):
    net.eval()

    y_pred_hot_list, y_pred_list = [], []
    for x in data.unsqueeze(dim=1):
        x = torch.tensor(x, dtype=torch.float, device=device)

        y_pred_hot = net.forward(x)  # 順伝播
        y_pred_hot = nn.functional.softmax(y_pred_hot, dim=1)
        
        y_pred = y_pred_hot.argmax(1).tolist()


        y_pred_hot_list.append(y_pred_hot.detach().cpu().numpy().flatten())
        y_pred_list.extend(y_pred)
    
    class_names = network.sample_names  # グラフの横軸（X軸）
    class_names.append('Others')
    count = [0] * (len(class_names))
    for y_pred, y_pred_hot in zip(y_pred_list, y_pred_hot_list):
        if y_pred_hot[y_pred] > 0.8:
            count[y_pred] += 1
        else:  # その他
            count[-1] += 1

    # ---- カスタマイズ設定 ----
    plt.rcParams.update({
        'font.family': 'Arial',   # フォントをArialに設定
        'axes.linewidth': 0.5,    # 軸の太さ
        'axes.edgecolor': 'black' # 軸の色を黒に設定
    })

    fig, ax = plt.subplots()

    # 縦軸に薄い灰色の補助線を追加（実線）
    ax.grid(True, axis='y', linestyle='-', color='lightgray', alpha=0.7, zorder=0)

    # 棒グラフの幅を狭く (デフォルトは0.8、ここでは0.4に設定)
    bar_width = 0.4
    indices = np.arange(len(class_names))  # 棒のX座標

    # 棒グラフの塗りつぶしを水色に戻す
    bars = ax.bar(indices, count, color='skyblue', edgecolor='black', linewidth=0.7, width=bar_width, zorder=3)

    for x, c in zip(indices, count):
        plt.text(x, c, str(c), ha='center', va='bottom', fontsize=10, zorder=4)

    # 軸の設定：上と右の枠線を非表示にし、左と下の枠線は黒で表示
    ax.spines['top'].set_visible(False)    # 上の枠線を非表示
    ax.spines['right'].set_visible(False)  # 右の枠線を非表示
    ax.spines['left'].set_linewidth(0.5)   # 左の軸線を細く表示
    ax.spines['left'].set_color('black')   # 左の軸線の色を黒に設定
    ax.spines['bottom'].set_linewidth(0.5) # 下の軸線を細く表示
    ax.spines['bottom'].set_color('black') # 下の軸線の色を黒に設定

    plt.xticks(indices, class_names, fontsize=10)  # X軸のラベル位置を調整
    plt.title('Classification Result', fontsize=14, fontweight='bold')  # タイトルを太字に
    plt.xlabel('Bacteria', fontsize=12, fontweight='bold')  # X軸ラベルを太字に
    plt.ylabel('Count', fontsize=12, fontweight='bold')  # Y軸ラベルを太字に
    plt.yticks(fontsize=10)
    
    plt.savefig(f'{data_dir}/classification_result_{measurement_freqs}.svg', format='svg')
    plt.show()
    plt.close()


def main():
    ext = data_file.split(".")[-1]
    if not ext == 'npy':
            
        if ext == 'txt': # txtファイルを選択した場合
            output_paths = function.reshape(data_path)
            for output_path in output_paths:
                if measurement_freqs in output_path:
                    csv_data_path = output_path
                    break
        
        else: # csvファイルを選択した場合
            csv_data_path = data_path
        if 'detrend' in csv_data_path:
            detrend_csv_data_path = csv_data_path
        elif 'merge' in csv_data_path:
            detrend_csv_data_path = csv_data_path
        else:
            detrend_csv_data_path = csv_data_path.replace('.csv', '_detrend.csv')
            df = pd.read_csv(csv_data_path, index_col=0)
            df = function.detrend(df)
            df.to_csv(detrend_csv_data_path)

        order1, order2 = network.order1, network.order2
        max_data_len = network.data_length
        data = function.peak_detect(detrend_csv_data_path, order1, order2, max_data_len, baseline_correct_flag=network.baseline_correct_flag)
    else:
        # npyファイルを選択した場合
        order1, order2 = network.order1, network.order2
        data = np.load(data_path)
        data[:,:,0]=data[:,:,0]/10**order1
        data[:,:,1]=data[:,:,1]/10**order2

    data = torch.tensor(data)
    data.unsqueeze(1)
    # data = np.load(r"C:\Users\Ken_Hayashida\東京工業大学・山本研 Dropbox\交流ナノポア\実験データ\測定データ\同軸リングフィルム\20240724\analysis_file\240731_121706\X.npy")

    # モデルを作成
    num_layers = info_json['model_params']['num_layers']
    num_filters = info_json['model_params']['num_filters']
    out_features = info_json['model_params']['out_features']
    dropout_rate = info_json['model_params']['dropout_rate']
    net = network.Network(num_layers=num_layers, num_filters=num_filters, out_features=out_features, dropout_rate=dropout_rate)
    net.to(device)

    # 保存された重みをロード
    net.load_state_dict(torch.load(f'{model_dir}/model.pth', torch.device('cpu')))

    evaluation(net, data)
    
    # network.pyを空のデータに戻す
    with open("network.py", "w") as f:
        f.write("# Network Souce")

if __name__ == '__main__':
    main()


