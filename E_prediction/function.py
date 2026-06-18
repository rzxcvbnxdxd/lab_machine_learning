import numpy as np
import pandas as pd
import os, sys, shutil, re
from typing import Union
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns
from pybaselines import Baseline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import optuna
import warnings
# optuna.visualization.matplotlibで表示されるwarningを非表示にする
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# reshape.pyと同様のもの
def reshape(input_path):
    ext = input_path.split(".")[-1]
    if ext == "txt" and re.search("[0-9]+k", input_path):
        print(f'{input_path} is being processed.')
        output_paths = []
        
        # txtファイルの読み込み
        f = open(input_path, 'r')
        data_list = f.readlines()
        
        # ヘッダーを取得
        count = 0
        for data in data_list:
            if data.startswith('%'):
                count += 1
            else:
                break
        
        header = data_list[3:count]
        header = [data.replace(' ', '').replace('%', '').replace('\n', '') for data in header]
        num_demod = int((count-3) / 4)        
        
        # ヘッダーの情報を取得
        demod_names = []
        index_columns, data_columns = [], []
        for i in range(num_demod*2):
            # demod名
            demod_name = re.findall('[A-Z][a-z]+[0-9]', header[i*2])[0]
            demod_names.append(demod_name)
            
            # 1,2列目のデータ情報
            index_column, data_column = header[i*2+1].split(',')
            index_columns.append(index_column)
            data_columns.append(data_column)
        
        # 1列目のデータが全て同じ（通常はTime(s)）であるか確認
        if len(set(index_columns)) == 1:
            index_column = index_columns[0]
        else:
            print('Wrong data')
            print('-'*100)
            return
        
        # txtファイルの読み込み
        df = pd.read_csv(input_path, sep=";", header=None, skiprows=count)
        df.columns = [index_column, "Data"]
        
        # 欠損値の確認
        if df.isnull().values.sum() != 0:
            print('This data includes NaN.')
            print('-'*100)
            return
        
        # データフレームの分割
        diff = df[index_column].diff() < 0
        end_indexes = diff[diff].index.tolist()
        if len(end_indexes) != num_demod*2 - 1:
            print('The number of data is insufficient for the number of measurement frequencies.')
            return
        df_list = []
        begin_index = 0
        for i, end_index in enumerate(end_indexes):
            df_tmp = df[begin_index : end_index].rename(columns={'Data': data_columns[i]})
            df_list.append(df_tmp)
            begin_index = end_index + 1
        # 最後のdfの作成
        df_tmp = df[begin_index:].rename(columns={'Data': data_columns[-1]})
        df_list.append(df_tmp) 
        
        # ファイル名から測定周波数を取得
        d_name, f_name = os.path.split(input_path)
        freq_list = re.findall("[0-9]+k", f_name)
        freq_list = freq_list[:num_demod]
        
        # データの結合
        select_result = None
        for i in range(1, 5):
            data_idxes = []
            for j in range(len(demod_names)):
                if demod_names[j][-1] == str(i):
                    data_idxes.append(j)
            if len(data_idxes) == 2:
                if "Amplitude" in data_columns[data_idxes[0]]:
                    select_result = 1
                elif "Amplitude" in data_columns[data_idxes[1]]:
                    select_result = 2
                else:
                    # 結合する順番を手動で指定
                    while True:
                        if select_result == None:
                            print(rf'{data_columns[data_idxes[0]]} {data_columns[data_idxes[1]]} --> 1')
                            print(rf'{data_columns[data_idxes[1]]} {data_columns[data_idxes[0]]} --> 2')
                            select_result = input('>>> ')
                        if select_result == str(1):
                            select_result = 1
                            break
                        elif select_result == str(2):
                            select_result = 2
                            break
                        else:
                            print('Your input was wrong. Try again.')
                            select_result = None
                            continue
                if select_result == 1:
                    df_merged = pd.merge(df_list[data_idxes[0]], df_list[data_idxes[1]], on=index_column, how="inner")
                elif select_result == 2:
                    df_merged = pd.merge(df_list[data_idxes[1]], df_list[data_idxes[0]], on=index_column, how="inner")
                
                df_merged = df_merged.set_index(index_column, drop=True)
                
                # 秒数が0から始まるよう調整
                df_merged.index += (-1) * df_merged.index[0]
                
                # データの出力
                output_path_v1 = rf"{d_name}\{freq_list[0]}.csv"
                output_path_v2 = rf"{d_name}\{freq_list[0]}_01.csv"
                if os.path.isfile(output_path_v1):
                    os.rename(output_path_v1, output_path_v2)
                    output_path = rf"{d_name}\{freq_list[0]}_02.csv"
                elif os.path.isfile(output_path_v2):
                    output_path = output_path_v2
                    i = 3
                    while os.path.isfile(output_path):
                        output_path = rf"{d_name}\{freq_list[0]}_{i:02}.csv"
                        i += 1
                else:
                    output_path = output_path_v1
                df_merged.to_csv(output_path, index=True)
                print(f"{output_path} was created.")
                output_paths.append(output_path)
                freq_list.pop(0)
        
        print('-'*100)
    return output_paths

# detrend.pyと同様のもの
def detrend_algo(times, data):
    baseline_fitter = Baseline(times, check_finite=False)
    detrend_data = data
    # 「ベースライン推定 -> トレンド除去」を繰り返す
    # トレンド除去 step１
    baseline, _ = baseline_fitter.pspline_iasls(detrend_data, spline_degree=1)
    detrend_data -= baseline
    # トレンド除去 step２
    time_length = times[-1] - times[0]
    if time_length > 30:
        baseline, _ = baseline_fitter.aspls(data, lam=1e15)
        detrend_data -= baseline
    else:
        pass
    # トレンド除去 step３
    baseline = np.full(len(detrend_data), detrend_data.median())
    detrend_data -= baseline
    return detrend_data

def detrend(df):
    columns = df.columns
    time = df.index
    data1 = df[columns[0]] # current
    data2 = df[columns[1]] # phase

    num1 = -1
    detrend_data = detrend_algo(time, data1*num1)
    df[f"Detrend{columns[0]}"] = detrend_data * num1
    df[f"Smoothed{columns[0]}"] = df[columns[0]] - df[f"Detrend{columns[0]}"]

    num2 = 1
    detrend_data = detrend_algo(time, data2*num2)
    df[f"Detrend{columns[1]}"] = detrend_data * num2
    df[f"Smoothed{columns[1]}"] = df[columns[1]] - df[f"Detrend{columns[1]}"]
    return df

class Peak:
    def __init__(self, data_arr, reverse_flag=False): # arr_dataはndarray
        self.data_arr = data_arr
        if reverse_flag:
            self.data_arr = data_arr * (-1)
        
        self.threshold = 0
        self.peak_idxes = []
        self.peak_values = []
        self.start_idxes = []
        self.end_idxes = []
    
    def auto_threshold(self) -> None:
        rms = np.sqrt(np.mean((self.data_arr[self.data_arr < 0])**2)) # 二乗平均平方根(実効値)を算出
        self.threshold = self.pdf * rms
    
    def manual_threshold(self) -> None:
        self.threshold = 0
        df = self.run_find_peaks(0)
        peak_idxes = df['PeakIndex'].tolist()
        peak_values = self.data_arr[peak_idxes]
        #グラフの作成
        plt.hist(peak_values, bins=200, alpha=0.5, color="blue")
        plt.show()
        self.threshold = float(input('Input threshold: '))
    
    def find_peaks(self, distance: int) -> pd.DataFrame:
        if type(self.threshold) is list:
            lower_threshold, upper_threshold = self.threshold
        else:
            lower_threshold = self.threshold
            upper_threshold = None
        i = 0
        data_len = len(self.data_arr)
        while True:
            if i >= data_len:
                break
            elif self.data_arr[i] < lower_threshold:
                i += 1
            else:
                start_idx = i
                data_temp = []
                while i < data_len and self.data_arr[i] >= lower_threshold:
                    data_temp.append(self.data_arr[i])
                    i += 1
                if i == start_idx + 1:
                    pass
                else:
                    end_idx = i - 1
                    max_value = max(data_temp)
                    max_idx = start_idx + data_temp.index(max_value)
                        
                    if len(self.end_idxes) > 0 and start_idx - self.end_idxes[-1] < distance: # ひとつ前のピークと同じ粒子由来のピークだった場合
                        if self.peak_values[-1] > max_value: # ひとつ前のピークの方が高い場合
                            pass
                        else:
                            self.peak_idxes[-1] = max_idx
                            self.peak_values[-1] = max_value
                        self.end_idxes[-1] = end_idx
                    elif end_idx - start_idx < distance: # ピーク幅がdistanceよりも小さい場合
                        pass
                    else:
                        self.peak_idxes.append(max_idx)
                        self.peak_values.append(max_value)
                        self.start_idxes.append(start_idx)
                        self.end_idxes.append(end_idx)
                i += 1
        
        df = pd.DataFrame(data={'PeakIndex':self.peak_idxes,'StartIndex':self.start_idxes, 'EndIndex':self.end_idxes, 'PeakValue':self.peak_values})
        if upper_threshold:
            df = df.query('PeakValue < @upper_threshold')
        self.peak_idxes = df['PeakIndex']
        self.start_idxes = df['StartIndex']
        self.end_idxes = df['EndIndex']
        self.peak_values = df['PeakValue']
        return df
    
    def remove_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        # ピーク幅に対して外れ値除去
        if len(df) < 5:
            pass
        else:
            df['Datasize'] = df['EndIndex'] - df['StartIndex']
            min_datasize, max_datasize = get_iqr(df['Datasize'].values)
            if min_datasize != max_datasize:
                df = df[(min_datasize < df['Datasize']) & (df['Datasize'] < max_datasize)]
            df = df.drop(columns='Datasize')
            
            # ピーク高さに対して外れ値除去
            min_value, max_value = get_iqr(df['PeakValue'].values)
            df = df[(min_value < df['PeakValue']) & (df['PeakValue'] < max_value)]
        return df
    
    def detect_peaks(self, threshold_mode, pdf, find_peaks_mode, distance) -> tuple[float, pd.DataFrame]:
        self.threshold_mode = threshold_mode
        self.pdf = pdf
        self.find_peaks_mode = find_peaks_mode
        if self.threshold_mode == 'auto':
            self.auto_threshold()
        elif self.threshold_mode == 'manual':
            self.manual_threshold()
        else:
            print('Invalid threshold_mode.')
        df = self.find_peaks(distance)
        df = self.remove_outlier(df)
        return self.threshold, df
    
def get_iqr(values: Union[list, np.ndarray], c: float = 1.5):
    q75, q25 = np.percentile(np.array(values), [75 ,25])
    iqr = q75 - q25
    min_value = q25 - iqr * c
    max_value = q75 + iqr * c
    return min_value, max_value

def matching_df(df1: pd.DataFrame, df2: pd.DataFrame) ->pd.DataFrame:
    peak_times1 = df1.index.values.tolist()
    peak_times2 = df2.index.values.tolist()
    
    start_column1 = [column for  column in df1.columns if 'StartTime' in column]
    end_column1 = [column for  column in df1.columns if 'EndTime' in column]
    start_column2 = [column for  column in df2.columns if 'StartTime' in column]
    end_column2 = [column for  column in df2.columns if 'EndTime' in column]
    
    start_times1 = df1[start_column1[0]].values
    end_times1 = df1[end_column1[0]].values
    start_times2 = df2[start_column2[0]].values
    end_times2 = df2[end_column2[0]].values
    
    j = 0
    for i, peak_time1 in enumerate(peak_times1):
        start_time1 = start_times1[i]
        end_time1 = end_times1[i]
        while j < len(df2):
            start_time2 = start_times2[j]
            end_time2 = end_times2[j]
            if start_time1 > end_time2: # ピーク2がピーク1より手前にある場合
                j += 1
                continue
            elif start_time2 > end_time1: # ピーク2がピーク1より後方にある場合
                break
            else:
                peak_time2 = peak_times2[j]
                df1.loc[peak_time1, 'num'] = i
                df2.loc[peak_time2, 'num'] = i
                j += 1
                break
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    if 'num' in df1.columns:
        df = pd.merge(df1, df2, how="inner", on="num").dropna()
        df = df.drop('num', axis=1)
    else:
        df = pd.DataFrame(columns=df1.columns.tolist())
    return df

def peak_detect(data_path, order1, order2, max_data_len, threshold_mode='auto', pdf=5, find_peaks_mode='normal', baseline_correct_flag=False):
    df = pd.read_csv(data_path, index_col=0)
    column1, column2 = df.columns.tolist()[:2]
    time = df.index.values.tolist()
    distance = int(0.002 / (time[1] - time[0])) # 隣り合うピークの間隔が0.002秒以内なら同じピークをみなす

    # current (voltage)
    data1 = df[column1].to_list()
    detrend_data1 = df[f'Detrend{column1}'].values
    peak = Peak(detrend_data1, reverse_flag=True)
    threshold1, df_peak1 = peak.detect_peaks(threshold_mode, pdf, find_peaks_mode, distance)
    df_peak1['PeakTime'] = [time[i] for i in df_peak1['PeakIndex'].tolist()]
    df_peak1['StartTime'] = [time[i] for i in df_peak1['StartIndex'].tolist()]
    df_peak1['EndTime'] = [time[i] for i in df_peak1['EndIndex'].tolist()]
    df_peak1['PeakValue'] = [detrend_data1[i] for i in df_peak1['PeakIndex'].tolist()]
    df_peak1['OriginalPeakValue'] = [data1[i] for i in df_peak1['PeakIndex'].tolist()]

    # phase
    data2 = df[column2].to_list()
    detrend_data2 = df[f'Detrend{column2}'].values
    peak = Peak(detrend_data2)
    threshold2, df_peak2 = peak.detect_peaks(threshold_mode, pdf, find_peaks_mode, distance)
    df_peak2['PeakTime'] = [time[i] for i in df_peak2['PeakIndex'].tolist()]
    df_peak2['StartTime'] = [time[i] for i in df_peak2['StartIndex'].tolist()]
    df_peak2['EndTime'] = [time[i] for i in df_peak2['EndIndex'].tolist()]
    df_peak2['OriginalPeakValue'] = [data2[i] for i in df_peak2['PeakIndex'].tolist()]

    df_peak = matching_df(df_peak1.add_prefix(f'{column1}_').set_index(f'{column1}_PeakTime'), df_peak2.add_prefix(f'{column2}_').set_index(f'{column2}_PeakTime'))

    if len(df_peak) > 0:
        start_idxes1 = df_peak.loc[:, f'{column1}_StartIndex'].tolist()
        end_idxes1 = df_peak.loc[:, f'{column1}_EndIndex'].tolist()
        start_idxes2 = df_peak.loc[:, f'{column2}_StartIndex'].tolist()
        end_idxes2 = df_peak.loc[:, f'{column2}_EndIndex'].tolist()
        
        df_detrend = df.loc[:, [f'Detrend{column1}', f'Detrend{column2}']]
        df_smoothed = df.loc[:, [f'Smoothed{column1}', f'Smoothed{column2}']]
        df_tmp_list = []
        if len(df_detrend) == 0:
            df_tmp_list = None
        else:
            for start_idx1, end_idx1, start_idx2, end_idx2 in zip(start_idxes1, end_idxes1, start_idxes2, end_idxes2):
                series_tmp1 = df_detrend.iloc[int(start_idx1):int(end_idx1)+1, 0].copy() # Amplitude
                series_tmp2 = df_detrend.iloc[int(start_idx2):int(end_idx2)+1, 1].copy() # Phase
                
                # ベースラインからの変化率を求める
                if baseline_correct_flag:
                    series_smoothed_tmp1 = df_smoothed.iloc[int(start_idx1):int(end_idx1)+1, 0].copy() # Amplitude
                    series_smoothed_tmp2 = df_smoothed.iloc[int(start_idx2):int(end_idx2)+1, 1].copy() # Phase
                    series_tmp1 /= series_smoothed_tmp1
                    series_tmp2 /= series_smoothed_tmp2
                df_tmp = pd.merge(series_tmp1, series_tmp2, how='outer', left_index=True, right_index=True)
                df_tmp = df_tmp.fillna(0) # 欠損値を0で補間
                df_tmp.reset_index(inplace=True, drop=True)
                df_tmp_list.append(df_tmp)
    else:
        print('Unable to create dataset due to insufficient number of peaks.')
        sys.exit()
    
    # DataFrameを切り取って保存
    X = []
    if df_peak is None: # ピークが1つも検出されなかった場合はスキップ
        print('Cannnot detecte peaks.')
        sys.exit()
    for df_tmp in df_tmp_list:
        df_tmp[f'Detrend{column1}'] /= 10**order1
        df_tmp[f'Detrend{column2}'] /= 10**order2
        if max_data_len >= len(df_tmp):
            nan_arr = np.zeros((max_data_len-len(df_tmp), 2))
            x = np.concatenate([df_tmp.values, nan_arr])
        else:
            x = df_tmp[:max_data_len].values
        X.append(x)
    
    return X

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
    accuracy = round(accuracy_score(y_true, y_pred), 3) 
    precision = round(precision_score(y_true, y_pred, average=average), 3) 
    recall = round(recall_score(y_true, y_pred, average=average), 3) 
    f1 = round(f1_score(y_true, y_pred, average=average), 3) 
    plt.title(f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1-measure:{f1}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.savefig(rf"{save_dir}\confusion_matrix.svg")
    plt.close()
    
    sns.heatmap(cm_ratio, annot=True, fmt='.2f', cmap='Blues', xticklabels=sample_names, yticklabels=sample_names, vmin=0, vmax=1)
    accuracy = round(accuracy_score(y_true, y_pred), 3) 
    precision = round(precision_score(y_true, y_pred, average=average), 3) 
    recall = round(recall_score(y_true, y_pred, average=average), 3) 
    f1 = round(f1_score(y_true, y_pred, average=average), 3) 
    plt.title(f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, F1-measure:{f1}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.savefig(rf"{save_dir}\confusion_matrix_v2.svg")
    plt.close()
    print(f"Accuracy: {accuracy}")
    return accuracy