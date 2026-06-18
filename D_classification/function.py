import pandas as pd
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import optuna
import warnings
# optuna.visualization.matplotlibで表示されるwarningを非表示にする
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

def make_dirs(path_list):
    val_type = type(path_list)
    if val_type == list:
        for path in path_list:
            if os.path.exists(path):
                    shutil.rmtree(path)
            else:
                pass
            os.makedirs(path)
    elif val_type == str:
        path = path_list
        if os.path.exists(path):
                shutil.rmtree(path)
        else:
            pass
        os.makedirs(path)

# 独自のround関数を定義
def my_round(number, ndigits=0):
    p = 10**ndigits
    return (number * p * 2 + 1) // 2 / p

def specificity_score(y_true, y_pred, sample_num):
    result = []
    for c in range(sample_num):
        specificity = recall_score(np.array(y_true)!=c, np.array(y_pred)!=c)
        result.append(specificity)
    return np.mean(result)

def make_confusion_matrix(y_true, y_pred, sample_names, save_dir: str) -> None:
    if len(set(y_true)) == 2:
        average = 'binary'
    else:
        average = 'macro'
        
    cm = confusion_matrix(y_true, y_pred)
    cm_ratio = cm / cm.sum(axis=1).reshape(-1, 1)
    cm_ratio = np.round(cm_ratio, 2)

    plt.figure()
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sample_names, yticklabels=sample_names)
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_percentile = my_round(accuracy*100, 2) # 正解率
    f1_percentile = my_round(f1_score(y_true, y_pred, average=average)*100, 2) # F値
    precision_percentile = my_round(precision_score(y_true, y_pred, average=average)*100, 2)  # 適合率
    recall_percentile = my_round(recall_score(y_true, y_pred, average=average)*100, 2) # 再現率(感度)
    specificity_percentile = my_round(specificity_score(y_true, y_pred, len(sample_names))*100, 2) # 特異度
    plt.title(f"Accuracy: {accuracy_percentile} % F1-measure: {f1_percentile} % Precision: {precision_percentile} %\nRecall(Sensitivity): {recall_percentile} % Specificity: {specificity_percentile} %")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.savefig(rf"{save_dir}\confusion_matrix.svg")
    plt.close()
    
    sns.heatmap(cm_ratio, annot=True, fmt='.2f', cmap='Blues', xticklabels=sample_names, yticklabels=sample_names, vmin=0, vmax=1)
    plt.title(f"Accuracy: {accuracy_percentile} % F1-measure: {f1_percentile} % Precision: {precision_percentile} %\nRecall(Sensitivity): {recall_percentile} % Specificity: {specificity_percentile} %")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.savefig(rf"{save_dir}\confusion_matrix_v2.svg")
    plt.close()
    print(f"Accuracy: {accuracy_percentile} %")
    return accuracy
    
def make_learning_curve(train_loss, valid_loss, save_dir: str) -> None:
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend(loc = 0)
    plt.savefig(rf"{save_dir}\learning_curve.svg")
    plt.close()

def make_optimize_result(studies, save_dir: str):
    if len(studies) > 1:
        # Empirical distribution function plot
        optuna.visualization.matplotlib.plot_edf(studies)
        plt.savefig(rf'{save_dir}\edf_all.svg')
        plt.close()

        # Optimization history plot
        optuna.visualization.matplotlib.plot_optimization_history(studies, error_bar=True)
        plt.savefig(rf'{save_dir}\optimization_history_all.svg')
        plt.close()

        # extract optimization history from each study
        dfs = []
        for study in studies:
             df = study.trials_dataframe()
             dfs.append(df)
        df = pd.concat(dfs)

        study = min(studies, key=lambda study: study.best_value)
        extension = "_best"
    else:
        study = studies[0]
        df = study.trials_dataframe()
        extension = ""
    
    # Hyperparameter importances
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f'{save_dir}\param_importance{extension}.svg')
    plt.close()
    
    # Empirical distribution function plot
    optuna.visualization.matplotlib.plot_edf(study)
    plt.savefig(rf'{save_dir}\edf{extension}.svg')
    plt.close()

    # Optimization history plot
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(rf'{save_dir}\optimization_history{extension}.svg')
    plt.close()

    # Parallel coordinate plot
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig(rf'{save_dir}\parallel_coordinate{extension}.svg')
    plt.close()

    # Slice plot
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(rf'{save_dir}\slice{extension}.svg')
    plt.close()
    
    # Optimization history csv
    df.to_csv(rf'{save_dir}\optimization_history.csv')
    
    return study