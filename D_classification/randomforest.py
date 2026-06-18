import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
from datetime import datetime
from scipy.stats import skew, kurtosis
import json
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
import optuna
import settings, function

path = settings.path

output_dir = rf"{path}\classification_file\randomforest\{datetime.now().strftime('%y%m%d_%H%M%S')}"
function.make_dirs(output_dir)

with open(rf'{path}\info.json', 'r', encoding='utf-8') as f:
    info_json = json.load(f)

sample_names = info_json['samples'].keys()

n_splits = 5

if settings.seed is not None:
    seed = settings.seed
else:
    seed = 42

rus = RandomUnderSampler(sampling_strategy='auto', random_state=seed)
ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def extract_features():
    X = np.load(rf'{path}/X.npy', allow_pickle=True)
    y = np.load(rf'{path}/y.npy', allow_pickle=True)
    max1_list, width1_list, area1_list, mean1_list, std1_list, skew1_list, kurt1_list = [], [], [], [], [], [], []
    max2_list, width2_list, area2_list, mean2_list, std2_list, skew2_list, kurt2_list = [], [], [], [], [], [], []
    for peak_data in X:
        # Amplitude
        peak_data1 = peak_data[:, 0]
        peak_data1 = peak_data1[peak_data1!=0] # 0を消去
        peak_data1 = peak_data1 * (-1)
        max1_list.append(max(peak_data1))
        width1_list.append(len(peak_data1))
        area1_list.append(sum(peak_data1))
        mean1_list.append(np.mean(peak_data1))
        std1_list.append(np.std(peak_data1))
        skew1_list.append(skew(peak_data1))
        kurt1_list.append(kurtosis(peak_data1))

        # Phase
        peak_data2 = peak_data[:, 1]
        peak_data2 = peak_data2[peak_data2!=0] # 0を消去
        max2_list.append(max(peak_data2))
        width2_list.append(len(peak_data2))
        area2_list.append(sum(peak_data2))
        mean2_list.append(np.mean(peak_data2))
        std2_list.append(np.std(peak_data2))
        skew2_list.append(skew(peak_data2))
        kurt2_list.append(kurtosis(peak_data2))
    X = pd.DataFrame({'max1': max1_list,
                      'width1': width1_list,
                      'area1': area1_list,
                      'mean1': mean1_list,
                      'std1': std1_list,
                      'skew1': skew1_list,
                      'kurt1': kurt1_list,
                      'max2': max2_list,
                      'width2': width2_list,
                      'area2': area2_list,
                      'mean2': mean2_list,
                      'std2': std2_list,
                      'skew2': skew2_list,
                      'kurt2': kurt2_list})
    return X, y

def train(X, y, n_estimators, max_depth, max_features):
    rf = RandomForestClassifier(random_state=seed,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                max_features=max_features,
                                class_weight='balanced')
    scoring = 'accuracy'
    output = cross_validate(rf, 
                            X,
                            y,
                            cv=skf, 
                            n_jobs =-1,
                            scoring=scoring,
                            verbose=0,
                            return_estimator=True)

    column_list = X.columns
    feature_importance = pd.Series(0, index=column_list, dtype='float64')
    for estimator in output['estimator']:
        feature_importance += estimator.feature_importances_

    df_result = pd.DataFrame(feature_importance/n_splits, index=column_list, columns=['importance'])
    df_result = df_result.sort_values('importance')
    left = [i for i in range(len(column_list))]
    plt.barh(left, df_result['importance'].tolist())
    plt.yticks(left, df_result.index)
    plt.savefig(rf"{output_dir}\feature_importance.svg")
    plt.close()

    y_pred = cross_val_predict(rf, X, y, cv=skf)

    accuracy = function.make_confusion_matrix(y, y_pred, sample_names, output_dir)

    return accuracy

def objective(trial):
    n_estimators =  trial.suggest_int('n_estimators', 1, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 1000)
    max_features = trial.suggest_categorical('max_features', ['sqrt','log2', None])
    
    X, y = extract_features()
    accuracy = train(X, y, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    return accuracy

def optimize(n_trials=100, study_num=1):
    studies = []
    for i in range(study_num):
        study = optuna.create_study(direction='maximize', study_name=f'study{i+1}', sampler=optuna.samplers.TPESampler(seed=seed+i))
        # study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=seed)) # ランダムサーチ
        study.optimize(objective, n_trials=n_trials)  # 試行回数を指定
        studies.append(study)
    
    # 最適化結果を可視化
    study = function.make_optimize_result(studies, output_dir)
    
    main(**study.best_params)

    
def main(n_estimators=100, max_depth=None, max_features='sqrt'):
    X, y = extract_features()
    train(X, y, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    X['class'] = y
    X.to_csv(rf'{output_dir}\feature_data.csv', index=False)

if __name__ == "__main__":
    if settings.trial_num:
        optimize(n_trials=settings.trial_num, study_num=settings.study_num)
    else:
        main()