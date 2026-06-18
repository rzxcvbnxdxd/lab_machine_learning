import numpy as np
import torch
import torch.nn as nn
import os, sys, json, shutil

# モデルディレクトリとX.npyのパスを指定
model_dir = r"C:\Users\ylab\Downloads\250327_新型チップ測定データ\250327_新型チップ測定データ\中間発表用解析\250821_123701\classification_file\cnn\250821_125241"
x_path = r"C:\Users\ylab\Downloads\250327_新型チップ測定データ\250327_新型チップ測定データ\中間発表用解析\250821_123701\X.npy"

# 分類モデルの情報を取得
shutil.copyfile(rf"{model_dir}\network.py", r"C:\Users\ylab\Desktop\研究プログラム\ingot-main\E_prediction\network.py")
import network

with open(f'{model_dir}/info.json', 'r', encoding='utf-8') as f:
    info_json = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    # X.npyの読み込み
    data = np.load(x_path)
    order1, order2 = network.order1, network.order2
    data[:,:,0] = data[:,:,0] / 10**order1
    data[:,:,1] = data[:,:,1] / 10**order2
    data = torch.tensor(data, dtype=torch.float, device=device)

    # モデルの構築
    num_layers = info_json['model_params']['num_layers']
    num_filters = info_json['model_params']['num_filters']
    out_features = info_json['model_params']['out_features']
    dropout_rate = info_json['model_params']['dropout_rate']
    net = network.Network(num_layers=num_layers, num_filters=num_filters, out_features=out_features, dropout_rate=dropout_rate)
    net.to(device)
    net.load_state_dict(torch.load(f'{model_dir}/model.pth', map_location=device))
    net.eval()

    # 予測
    y_pred_list = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0)  # バッチ次元追加
            y_pred_hot = net(x)
            y_pred = y_pred_hot.argmax(1).item()
            y_pred_list.append(y_pred)

    # z.npyとして保存（X.npyと同じディレクトリ）
    z_path = os.path.join(os.path.dirname(x_path), "z.npy")
    np.save(z_path, np.array(y_pred_list))
    print(f"Prediction results saved to {z_path}")

if __name__ == '__main__':
    main()