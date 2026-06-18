import numpy as np
import random, json, inspect, shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import optuna
import torch, torchinfo
import torch.nn as nn

import settings
import function

path = settings.path

output_dir = rf"{path}\classification_file\cnn\{datetime.now().strftime('%y%m%d_%H%M%S')}"
function.make_dirs(output_dir)

shutil.copyfile(r"C:\Users\rzxcv\OneDrive\Desktop\研究\kenkyu\D_classification\settings.py", rf"{output_dir}\settings.py")

json_path = rf'{path}\info.json'
with open(json_path, 'r', encoding='utf-8') as f:
    info_json = json.load(f)

num_classes = info_json['dataset_info']['num_classes']
data_length = info_json['dataset_info']['data_length']
baseline_correct_flag = info_json['dataset_info']['baseline_correct_flag']
measurement_freqs = info_json['dataset_info']['measurement_freqs']
order1 = info_json['dataset_info']['order1']
order2 = info_json['dataset_info']['order2']

feature_num = len(measurement_freqs) * 2

sample_names = list(info_json['samples'].keys())
sample_paths = [info_json['samples'][sample_name]['path'] for sample_name in sample_names]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

if settings.seed is not None:
    seed = settings.seed
else:
    seed = 42

# seedの固定
def fix_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float, device=device), torch.tensor(self.y[idx], dtype=torch.long, device=device)

class Network(nn.Module):
    def __init__(self, num_layers=6, num_filters=64, out_features=256, dropout_rate=0.5):
        super(Network, self).__init__()
        modules = []
        features = data_length
        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Conv1d(feature_num, num_filters, kernel_size=3, stride=1))
            else:
                modules.append(nn.Conv1d(num_filters, num_filters, kernel_size=3, stride=1))
            modules.append(nn.BatchNorm1d(num_filters))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            features = (features - 2) // 2
            if features <= 3: # input size は kernel size より大きい必要がある	
                print(f"num_layers is {i}")
                break
        
        modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Flatten())
        modules.append(nn.Linear(num_filters*features, out_features))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(out_features, num_classes))
        self.model = nn.Sequential(*modules)
        
    def forward(self, X):
        X = X.permute(0, 2, 1) # (バッチサイズ, 時系列長さ, 特徴量数) -> (バッチサイズ, 特徴量数, 時系列長さ)
        X = self.model(X)  # Sequentialモデルで順伝播
        return X

def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)
        m.to(device)

def calculate_class_weights(y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    return class_weights

def data_load(batch_size):
    X = np.load(rf'{path}\X.npy', allow_pickle=True)
    # 電流・位相データの値のオーダーを揃える
    X[:, :, 0] /= 10**order1
    X[:, :, 1] /= 10**order2
    y = np.load(rf'{path}\y.npy', allow_pickle=True)
    
    if settings.selected_peak_idx is not None:
        # 指定範囲のデータを抽出
        selected_X = []
        selected_y = []

        for sample_class, (start_idx, end_idx) in settings.selected_peak_idx.items():
            # 該当クラスのデータインデックスを取得
            class_idxes = np.where(y == sample_class)[0]
            # 範囲内のデータを選択
            selected_indices = class_idxes[start_idx:end_idx]
            selected_X.append(X[selected_indices])
            selected_y.append(y[selected_indices])

        # リストを連結して新しいデータセットを作成
        X = np.concatenate(selected_X, axis=0)
        y = np.concatenate(selected_y, axis=0)
    
    data_num = len(y)
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, test_size=int(data_num*0.15), random_state=seed, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid, test_size=int(data_num*0.15), random_state=seed, stratify=y_trainvalid)
    
    global class_weights
    # クラスごとの重みを計算
    class_weights = calculate_class_weights(y_test)
    
    train_data = dataset(X_train, y_train)
    valid_data = dataset(X_valid, y_valid)
    test_data = dataset(X_test, y_test)
    
    global train_data_num, vallid_data_num, test_data_num
    train_data_num = len(train_data)
    vallid_data_num = len(valid_data)
    test_data_num = len(test_data)
    
    global dataloader_train, dataloader_valid, dataloader_test
    
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size) # shuffle=Trueにしてしまうと再現性が保たれないので注意
    dataloader_valid = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    dataloader_test = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

def train(net, optimizer, patience=10):
    # 損失関数を定義
    loss_function = nn.CrossEntropyLoss(weight=class_weights) # クラスごとのデータ数に応じで重みを付ける

    best_valid_loss = float('inf')  # 最良の検証損失を保持する変数を初期化
    counter = 0  # カウンターを初期化
    train_loss_list, valid_loss_list = [], []
    acc_val_list = []
    epochs = 1
    while True:
        losses_train, losses_valid = [], []

        net.train()
        n_train = 0
        acc_train = 0
        for X, y in dataloader_train:
            n_train += y.size()[0]

            net.zero_grad()  # 勾配の初期化
            
            X = X.to(device)  # テンソルをGPUに移動

            y_hot = torch.eye(num_classes, device=device)[y]  # 正解ラベルをone-hot vector化

            y = y.to(device)
            y_hot = y_hot.to(device)  # 正解ラベルとone-hot vectorをそれぞれGPUに移動

            y_pred_hot = net.forward(X)  # 順伝播

            loss = loss_function(y_pred_hot, y_hot)  # 誤差(クロスエントロピー誤差関数)の計算

            loss.backward()  # 誤差の逆伝播

            optimizer.step()  # パラメータの更新

            y_pred = y_pred_hot.argmax(1)  # 最大値を取るラベルを予測ラベルとする

            acc_train += (y_pred == y).float().sum().item()
            losses_train.append(loss.tolist())

        train_loss = np.mean(losses_train)
        train_loss_list.append(train_loss)
        acc_train /= n_train
        
        net.eval()
        n_val = 0
        acc_val = 0
        for X, y in dataloader_valid:
            n_val += y.size()[0]

            X = X.to(device)  # テンソルをGPUに移動

            y_hot = torch.eye(num_classes, device=device)[y]  # 正解ラベルをone-hot vector化

            y = y.to(device)
            y_hot = y_hot.to(device)  # 正解ラベルとone-hot vectorをそれぞれGPUに移動

            y_pred_hot = net.forward(X)  # 順伝播

            loss = loss_function(y_pred_hot, y_hot)  # 誤差(クロスエントロピー誤差関数)の計算

            y_pred = y_pred_hot.argmax(1)  # 最大値を取るラベルを予測ラベルとする

            acc_val += (y_pred == y).float().sum().item()
            losses_valid.append(loss.tolist())

        valid_loss = np.mean(losses_valid)  # 検証損失を計算
        valid_loss_list.append(valid_loss)
        
        acc_val /= n_val
        acc_val_list.append(acc_val)
        
        print(f'EPOCH: {epochs}, Train [Loss: {train_loss:.3f}, Accuracy: {acc_train:.3f}], Valid [Loss: {valid_loss:.3f}, Accuracy: {acc_val:.3f}]')

        # early stopping
        if settings.max_epoch and epochs >= settings.max_epoch:
            break
        else:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                counter = 0  # カウンターをリセット
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break
            else:
                epochs += 1
    
    function.make_learning_curve(train_loss_list, valid_loss_list, output_dir)
    
    return min(valid_loss_list), max(acc_val_list), epochs

def evaluation(net):
    net.eval()

    y_list, y_pred_list = [], []
    for X, y in dataloader_test:
        X = X.to(device)  # テンソルをGPUに移動

        y_pred_hot = net.forward(X)  # 順伝播
        y_pred = y_pred_hot.argmax(1).tolist()
        
        y_list.extend(y.tolist())
        y_pred_list.extend(y_pred)
    
    function.make_confusion_matrix(y_list, y_pred_list, sample_names, output_dir)

def objective(trial):
    # batch_size
    if settings.batch_size is not None:
        batch_size = settings.batch_size
    else:
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    
    # learning_rate
    if settings.learning_rate is not None:
        learning_rate = settings.learning_rate
    else:
        learning_rate = trial.suggest_categorical('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    
    # weight_decay
    if settings.weight_decay is not None:
        weight_decay = settings.weight_decay
    else:
        weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    
    # num_layers
    if settings.num_layers is not None:
        num_layers = settings.num_layers
    else:
        num_layers = trial.suggest_int('num_layers', 1, 9)
    
    # num_filters
    if settings.num_filters is not None:
        num_filters = settings.num_filters
    else:
        num_filters = trial.suggest_categorical('num_filters', [32, 64, 128, 256])
    
    # out_features
    if settings.out_features is not None:
        out_features = settings.out_features
    else:
        out_features = trial.suggest_categorical('out_features', [128, 256, 512, 1024])
        
    # dropout_rate
    if settings.dropout_rate is not None:
        dropout_rate = settings.dropout_rate
    else:
        dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    fix_seed()
    data_load(batch_size)
    net = Network(num_layers=num_layers, num_filters=num_filters, out_features=out_features, dropout_rate=dropout_rate)
    net.apply(init_weights)
    net.to(device)
    # DataParallelによる並列化
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    valid_loss, _, _ = train(net, optimizer)
    
    return valid_loss

def optimize(n_trials=100, study_num=1):
    studies = []
    for i in range(study_num):
        study = optuna.create_study(direction='minimize', study_name=f'study{i+1}', sampler=optuna.samplers.TPESampler(seed=seed+i))
        # study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=seed)) # ランダムサーチ
        study.optimize(objective, n_trials=n_trials)  # 試行回数を指定
        studies.append(study)
    
    # 最適化結果を可視化
    study = function.make_optimize_result(studies, output_dir)
    
    main(**study.best_params)

def main(batch_size=64, learning_rate=0.0001, weight_decay=0, num_layers=3, num_filters=64, out_features=128, dropout_rate=0.5):
    fix_seed()
    data_load(batch_size)
    net = Network(num_layers=num_layers, num_filters=num_filters, out_features=out_features, dropout_rate=dropout_rate)
    net.apply(init_weights)
    
    net.to(device)
    # DataParallelによる並列化
    dp_flag = False
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        dp_flag = True
    
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _, _, epochs = train(net, optimizer)

    # モデルを保存するパス
    model_path = rf"{output_dir}\model.pth"

    # モデルを保存
    if dp_flag: 
        torch.save(net.module.state_dict(), model_path)
    else:
        torch.save(net.state_dict(), model_path)

    # モデルを作成
    net = Network(num_layers=num_layers, num_filters=num_filters, out_features=out_features, dropout_rate=dropout_rate)
    net.to(device)

    # 保存された重みをロード
    net.load_state_dict(torch.load(model_path))

    evaluation(net)
    
    # モデルの情報を取得
    model_info = torchinfo.summary(net.to('cpu'), input_size=(batch_size, data_length, feature_num))

    # テキストファイルに保存
    output_file = rf"{output_dir}\model_info.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(str(model_info))
    
    params_json = {
        'data_num': {
            'train': train_data_num,
            'validation': vallid_data_num,
            'test': test_data_num
        },
        'model_params': {
            'num_layers': num_layers,
            'num_filters': num_filters,
            'out_features': out_features,
            'dropout_rate': dropout_rate
        },
        'algorithm_params': {
            'trial_num': settings.trial_num,
            'study_num': settings.study_num,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
    }
    
    info_json.update(params_json)
    
    with open(rf'{output_dir}\info.json', 'w', encoding="utf-8") as f:
        json.dump(info_json, f, ensure_ascii=False, indent=4)
    
    # ネットワークの保存	
    network_souce = inspect.getsource(Network)	
    with open(rf'{output_dir}/network.py', "w", newline="\n") as f:	
        f.write("# Network Souce\n\n")
        f.write('import torch.nn as nn\n\n')
        f.write(f'num_classes = {num_classes}\n')
        f.write(f'data_length = {data_length}\n')
        f.write(f'baseline_correct_flag = {baseline_correct_flag}\n')
        f.write(f'order1 = {order1}\n')
        f.write(f'order2 = {order2}\n')
        f.write(f'sample_names = {sample_names}\n')
        f.write(f'feature_num={feature_num}\n')
        f.write(network_souce)

if __name__ == "__main__":
    if settings.trial_num:
        optimize(n_trials=settings.trial_num, study_num=settings.study_num)
    else:
        params = {
            'batch_size':settings.batch_size,
            'learning_rate': settings.learning_rate,
            'weight_decay': settings.weight_decay,
            'num_layers': settings.num_layers,
            'num_filters': settings.num_filters,
            'out_features': settings.out_features,
            'dropout_rate': settings.dropout_rate
        }
        params = {key: value for key, value in params.items() if value is not None} # valueがNoneのものを消去
        main(**params)