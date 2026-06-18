# Network Souce

import torch.nn as nn

num_classes = 7
data_length = 2000
baseline_correct_flag = True
order1 = -5
order2 = -5
sample_names = ['Deng4', 'Deng3', 'Deng2', 'Deng1', 'FluA', 'FluA_AICHI', 'FluA_UDORN']
feature_num=2
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
