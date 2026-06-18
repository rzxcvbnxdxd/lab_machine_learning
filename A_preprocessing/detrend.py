import pandas as pd
import numpy as np
from pybaselines import Baseline
import settings

path_list = settings.detrend_path
start_times = settings.start_times
end_times = settings.end_times

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
        baseline, _ = baseline_fitter.aspls(detrend_data, lam=1e15)
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

def main(df):
    if len(start_times) == 0:
        df = detrend(df)
        return df
    elif len(start_times) == len(end_times):
        data_length = len(df)
        df['number'] = np.arange(data_length) # cut前のインデックスを残しておきたいので新しく列を追加
        numbers = [data_length+1] # 各ブロックの1行目のnumberを収納するためのリスト
        df_cut = df.copy()
        for start, end in zip(reversed(start_times), reversed(end_times)):
            if end == -1:
                df_cut = df_cut[:start].copy()
            elif start > end:
                print('\nCutting start time is later than end time')
                return
            elif start == 0:
                df_cut = df_cut.shift(-1 * len(df_cut[:end])).dropna()
            else:
                df_top = df_cut[:start].copy()
                df_bottom = df_cut[start:].copy()
                df_bottom = df_bottom.shift(-1 * len(df_cut[start:end])).dropna()
                numbers.append(df_bottom.iat[0, -1])
                df_cut = pd.concat([df_top, df_bottom])
        numbers.append(df_cut.iat[0, -1])
        dfs = []
        numbers.reverse()
        for i in range(len(numbers)-1):
            start_num, end_num = numbers[i], numbers[i+1]
            df_tmp = df_cut.query('@start_num <= number < @end_num')
            df_tmp = df_tmp.drop(columns='number')
            df_tmp = detrend(df_tmp)
            dfs.append(df_tmp)
        df = pd.concat(dfs)
        return df
    else:
        # start_timesとend_timesの個数が違う時
        print('\nThe number of start_times and end_times are different.')
        return

if __name__ == '__main__':
    if isinstance(path_list, str):
        path_list = [path_list]
    for path in path_list:
        print(f'\r{path} is being processed.', end='')
        df = pd.read_csv(path, index_col=0)
        df = main(df)
        output_path = path.replace('.csv', '_detrend.csv')
        df.to_csv(output_path)
        print(f'\r{output_path} was created.')