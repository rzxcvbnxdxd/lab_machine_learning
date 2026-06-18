import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil, modf
from typing import Union
from pybaselines import Baseline
from scipy.signal import find_peaks, peak_widths
import os, shutil, jsonschema

def json_check(data, schema) -> bool:
    # Schemaに従ってバリデーションを行う
    try:
        jsonschema.validate(instance=data, schema=schema)
        return False
    except jsonschema.exceptions.ValidationError as e:
        print("Invalid JSON: " + e.message)
        return True

def make_dirs(path_list) -> None:
    val_type = type(path_list)
    if val_type != list:
        path_list = [path_list]
    for path in path_list:
        if os.path.exists(path):
                shutil.rmtree(path)
        else:
            pass
        os.makedirs(path)

def detrend(df) -> pd.DataFrame:
    columns = df.columns.values
    times = df.index
    baseline_fitter = Baseline(times, check_finite=False)
    for i, column in enumerate(columns):
        tmp = (-1)**(i+1)
        detrend_data = df[column] * tmp
        baseline, _ = baseline_fitter.pspline_iasls(detrend_data, spline_degree=1)
        detrend_data -= baseline
        baseline, _ = baseline_fitter.aspls(detrend_data, lam=1e15)
        detrend_data -= baseline
        baseline = np.full(len(detrend_data), detrend_data.median())
        detrend_data -= baseline
        detrend_data = detrend_data * tmp
        df[f'Smoothed{column}'] = df[column] - detrend_data
        df[f'Detrend{column}'] = detrend_data
    return df

class Peak:
    def __init__(self, arr_data, reverse_flag=False): # arr_dataはndarray
        self.arr_data = arr_data
        if reverse_flag:
            self.arr_data = arr_data * (-1)
        
        self.threshold = 0
        self.peak_idxes = []
        self.peak_values = []
        self.start_idxes = []
        self.end_idxes = []
    
    def auto_threshold(self) -> None:
        rms = np.sqrt(np.mean((self.arr_data[self.arr_data < 0])**2)) # 二乗平均平方根(実効値)を算出
        self.threshold = self.pdf * rms
    
    def manual_threshold(self) -> None:
        self.threshold = 0
        df = self.run_find_peaks(0)
        peak_idxes = df['PeakIndex'].tolist()
        peak_values = self.arr_data[peak_idxes]
        #グラフの作成
        plt.hist(peak_values, bins=200, alpha=0.5, color="blue")
        plt.show()
        self.threshold = float(input('Input threshold: '))

    def scipy_find_peaks(self, distance: int) -> pd.DataFrame:
        self.peak_idxes, properties = find_peaks(self.arr_data, height=self.threshold, distance=distance)
        self.peak_values = properties['peak_heights']
        _, _, start_ips, end_ips = peak_widths(self.arr_data, self.peak_idxes, rel_height=0.5)
        self.start_idxes = [ceil(start_ip) for start_ip in start_ips]
        self.end_idxes = [floor(end_ip) for end_ip in end_ips]

        df = pd.DataFrame({'PeakIndex':self.peak_idxes, 'StartIndex':self.start_idxes, 'EndIndex':self.end_idxes, 'PeakValue':self.peak_values})
        return df
    
    def normal_find_peaks(self, distance: int) -> pd.DataFrame:
        if type(self.threshold) is list:
            lower_threshold, upper_threshold = self.threshold
        else:
            lower_threshold = self.threshold
            upper_threshold = None
        i = 0
        data_len = len(self.arr_data)
        while True:
            if i >= data_len:
                break
            elif self.arr_data[i] < lower_threshold:
                i += 1
            else:
                start_idx = i
                data_temp = []
                while i < data_len and self.arr_data[i] >= lower_threshold:
                    data_temp.append(self.arr_data[i])
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
        
        df = pd.DataFrame({'PeakIndex':self.peak_idxes,'StartIndex':self.start_idxes, 'EndIndex':self.end_idxes, 'PeakValue':self.peak_values})
        if upper_threshold:
            df = df.query('PeakValue < @upper_threshold')
        self.peak_idxes = df['PeakIndex']
        self.start_idxes = df['StartIndex']
        self.end_idxes = df['EndIndex']
        self.peak_values = df['PeakValue']
        return df
    
    def run_find_peaks(self, distance: int) -> pd.DataFrame:
        self.distance = distance
        if self.find_peaks_mode == 'scipy':
            df = self.scipy_find_peaks(distance)
        else:
            df = self.normal_find_peaks(distance)
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
        df = self.run_find_peaks(distance)
        df = self.remove_outlier(df)
        return self.threshold, df

def ips_to_values(data: list, ips) -> list:
    values = []
    for ip in ips:
        decimal, integer = modf(ip)
        integer = int(integer)
        if integer+1 == len(data):
            value = data[integer] + (data[integer]-data[integer-1])*decimal
        else:
            value = data[integer] + (data[integer+1]-data[integer])*decimal
        values.append(value)
    return values

def get_iqr(values: Union[list, np.ndarray], c: float =1.5):
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

def get_order(value):
    value = str(value)
    if 'e' in value: # 指数表記ならば
        order = int(value.split('e')[-1])
    else:
        value_int, value_dec = value.split('.') # 整数部と小数部に分割
        if int(value_int) == 0: # 整数部が0ならば
            order = (value_dec.count('0')+1) * (-1) # 小数部からオーダーを取得
        else:
            order = len(value_int) - 1 # 整数部からオーダーを取得
    return order