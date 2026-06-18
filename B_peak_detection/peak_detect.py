import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import re, os, shutil, sys, statistics, json
import function, settings

sample_names = list(settings.samples.keys())
sample_paths = list(settings.samples.values())

num_classes = len(sample_names)

'''
パスに関する変数名の確認
output_path: 結果を保存するためのファイル
output_dir: プログラムを実行するたびに作られるファイル
output_dirs: output_dirの下に作られる、サンプル名のついたファイルのリスト
'''

def get_setting():
    error_flag = False
    # Check mesurement frequency
    sample_dirs, sample_files = [], []
    for sample_path in sample_paths:
        sample_dir, sample_file = os.path.split(sample_path)
        sample_dirs.append(sample_dir)
        sample_files.append(sample_file)
    
    measurement_freqs = []
    for sample_file in sample_files:
        freq = re.search(r'\d+k', sample_file)
        if freq is not  None:
            measurement_freqs.append(freq.group())
        else:
            print('Need to include the measurement frequency in the file name')
            sys.exit()
    if len(set(measurement_freqs)) == 1:
        global measurement_freq
        measurement_freq = measurement_freqs[0]
    else:
        print('Some data has different frequency.')
        sys.exit()
    
    # Check file existence
    if settings.multi_freqs:
        settings.multi_freqs = [f'{freq}k' for freq in settings.multi_freqs if 'k' not in str(freq)]
        global multi_freqs_sample_paths
        multi_freqs_sample_paths = [] # 他の周波数で測定したcsvファイルのパスを格納するためのリスト
        for sample_dir, sample_file in zip(sample_dirs, sample_files):
            multi_freqs_sample_files = [sample_file.replace(measurement_freq, freq, 1) for freq in settings.multi_freqs] # ファイル名の周波数部分(~k)の値を置換
            multi_freqs_sample_paths_tmp = [rf'{sample_dir}\{multi_freqs_sample_file}' for multi_freqs_sample_file in multi_freqs_sample_files]
            multi_freqs_sample_paths.append(multi_freqs_sample_paths_tmp)
        all_sample_paths = sum(multi_freqs_sample_paths, sample_paths)
    else:
        all_sample_paths = sample_paths
    for sample_path in all_sample_paths:
        if not os.path.isfile(sample_path):
            print(f"Not exist {sample_path}")
            error_flag = True
        else:
            pass

    output_path = settings.output_path
    if output_path == None:
        commonpath = os.path.commonpath(sample_paths)
        if os.path.isfile(commonpath):
            commonpath = os.path.split(commonpath)[0]
        output_path = rf'{commonpath}\analysis_file'
    else:
        pass
    global output_dir, settings_json

    # JSON Schemaに従ってバリデーションを行う
    settings_schema_json_path = r"C:\Users\rzxcv\OneDrive\Desktop\研究\kenkyu\B_peak_detection\settings_schema.json"
    with open(settings_schema_json_path, 'r') as f:
        settings_schema_json = json.load(f)
    settings_json = settings.params
    error_flag = function.json_check(settings_json, settings_schema_json)

    if error_flag:
        sys.exit()
    else:
        pass
    
    output_dir = rf"{output_path}\{datetime.now().strftime('%y%m%d_%H%M%S')}"
    print(f'output directory: {output_dir}')
    function.make_dirs(output_dir)
    shutil.copyfile(r"C:\Users\rzxcv\OneDrive\Desktop\研究\kenkyu\B_peak_detection\settings.py", rf"{output_dir}/settings.py")
    
    # JSONの設定を反映する
    global output_dirs, pdf, threshold_mode, thresholds, find_peaks_mode

    output_dirs = [rf"{output_dir}\{sample_name}" for sample_name in sample_names]
    function.make_dirs(output_dirs)
    threshold_mode = settings_json['find_peaks']['threshold_mode']
    if threshold_mode == 'auto':
        pdf = settings_json['find_peaks']['pdf']
        thresholds = [0] * len(sample_names)
    else:
        thresholds = settings_json['find_peaks'][thresholds]
    find_peaks_mode = settings_json['find_peaks']['find_peaks_mode']

def peak_detect():
    global df_list, df_peak_list, column1, column2
    df_list, df_peak_list = [], []
    for sample_path in sample_paths:
        df = pd.read_csv(sample_path, index_col=0)
        df_list.append(df)
    column1, column2 = df.columns.tolist()[:2]
    peak1_nums, peak2_nums, peak_matching_nums = [], [], []
    thresholds1, thresholds2 = [], []
    measurement_times = []
    for df, path in tqdm(zip(df_list, output_dirs), desc='peak detect', total=num_classes):
        time = df.index.values.tolist()
        distance = int(0.002 / (time[1] - time[0])) # 隣り合うピークの間隔が0.002秒以内なら同じピークとみなす
        measurement_times.append(time[-1]-time[0])

        # current (voltage)
        data1 = df[column1].to_list()
        detrend_data1 = df[f'Detrend{column1}'].values
        peak = function.Peak(detrend_data1, reverse_flag=True)
        threshold1, df_peak1 = peak.detect_peaks(threshold_mode, pdf, find_peaks_mode, distance)
        thresholds1.append(threshold1)
        df_peak1['PeakTime'] = [time[i] for i in df_peak1['PeakIndex'].tolist()]
        df_peak1['StartTime'] = [time[i] for i in df_peak1['StartIndex'].tolist()]
        df_peak1['EndTime'] = [time[i] for i in df_peak1['EndIndex'].tolist()]
        df_peak1['PeakValue'] = [detrend_data1[i] for i in df_peak1['PeakIndex'].tolist()]
        df_peak1['OriginalPeakValue'] = [data1[i] for i in df_peak1['PeakIndex'].tolist()]

        # phase
        data2 = df[column2].to_list()
        detrend_data2 = df[f'Detrend{column2}'].values
        peak = function.Peak(detrend_data2)
        threshold2, df_peak2 = peak.detect_peaks(threshold_mode, pdf, find_peaks_mode, distance)
        thresholds2.append(threshold2)
        df_peak2['PeakTime'] = [time[i] for i in df_peak2['PeakIndex'].tolist()]
        df_peak2['StartTime'] = [time[i] for i in df_peak2['StartIndex'].tolist()]
        df_peak2['EndTime'] = [time[i] for i in df_peak2['EndIndex'].tolist()]
        df_peak2['OriginalPeakValue'] = [data2[i] for i in df_peak2['PeakIndex'].tolist()]

        # csvに出力
        df_peak1.to_csv(rf"{path}\peak_{column1}_only.csv", index=False)
        df_peak2.to_csv(rf"{path}\peak_{column2}_only.csv", index=False)
        df_temp_matching = function.matching_df(df_peak1.add_prefix(f'{column1}_').set_index(f'{column1}_PeakTime'), df_peak2.add_prefix(f'{column2}_').set_index(f'{column2}_PeakTime'))
        df_temp_matching.to_csv(rf"{path}\peak.csv", index=False)
        df_peak_list.append(df_temp_matching)

        # ピーク数を記録
        peak1_nums.append(len(df_peak1))
        peak2_nums.append(len(df_peak2))
        peak_matching_nums.append(len(df_temp_matching))

    # jsonにthresholdを追加
    sample_info = {}
    for sample_name, sample_path, measurement_time, peak1_num, peak2_num, peak_matching_num, threshold1, threshold2 in zip(sample_names, sample_paths, measurement_times, peak1_nums, peak2_nums, peak_matching_nums, thresholds1, thresholds2):
        sample_info[sample_name] = {
                'path': os.path.normpath(sample_path),
                'measurement_time': measurement_time,
                'peak number': {
                    column1: peak1_num,
                    column2: peak2_num,
                    'both': peak_matching_num,
                },
                'threshold': {
                    column1: threshold1,
                    column2: threshold2
                }
            }
    settings_json['samples'] = sample_info

def make_time_series_data():
    df_tmp_list_list = []
    order1_list, order2_list = [], [] # 学習時、値のオーダーをそろえるために、位相・電流それぞれのオーダーを取得しておく
    TARGET_LEN = 2000 # 出力するタイムシリーズ長（ピークを中央に配置）
    max_data_len = TARGET_LEN # 固定長にする

    def _extract_centered_segment(series, start_idx, end_idx, peak_idx, target_len):
        """ series: pd.Series（元データ全体）、start_idx/end_idx: セグメントの絶対インデックス、
            peak_idx: 絶対ピーク位置。セグメント全体を取り出して、ピークが中央に来るよう target_len の配列に配置。
            範囲外はゼロでパディング。 """
        # セグメントを取得（相対インデックスにリセット）
        seg = series.iloc[int(start_idx):int(end_idx)+1].reset_index(drop=True)
        seg_len = len(seg)
        center = target_len // 2
        peak_rel = int(peak_idx) - int(start_idx)  # セグメント内でのピーク位置（0-origin）

        out = np.zeros(target_len, dtype=float)
        dst_start = center - peak_rel
        dst_end = dst_start + seg_len

        src_start = 0
        src_end = seg_len

        # 前にはみ出す場合
        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        # 後ろにはみ出す場合
        if dst_end > target_len:
            src_end = seg_len - (dst_end - target_len)
            dst_end = target_len

        if src_start < src_end:
            out[dst_start:dst_end] = seg.iloc[src_start:src_end].values

        return pd.Series(out).rename(series.name)

    for i, (df, df_peak) in tqdm(enumerate(zip(df_list, df_peak_list)), desc='make datasets', total=num_classes):
        if len(df_peak) > 0:
            start_idxes1 = df_peak[f'{column1}_StartIndex'].values
            end_idxes1 = df_peak[f'{column1}_EndIndex'].values
            peak_idxes1 = df_peak[f'{column1}_PeakIndex'].values
            start_idxes2 = df_peak[f'{column2}_StartIndex'].values
            end_idxes2 = df_peak[f'{column2}_EndIndex'].values
            peak_idxes2 = df_peak[f'{column2}_PeakIndex'].values
            
            df_tmp_list = []
            if len(df) == 0:
                df_tmp_list = None
            else:
                # ピーク部分を切り取る（ピークがウィンドウ中央に来るように固定長で切り出す）
                base_df_list = [df]
                measurement_freqs = [measurement_freq]
                if settings.multi_freqs:
                    for multi_freqs_sample_path in multi_freqs_sample_paths[i]:
                        base_df_list.append(pd.read_csv(multi_freqs_sample_path, index_col=0))
                    measurement_freqs += settings.multi_freqs

                for k, (start_idx1, end_idx1, start_idx2, end_idx2) in enumerate(zip(start_idxes1, end_idxes1, start_idxes2, end_idxes2)):
                    peak_idx1 = int(peak_idxes1[k])
                    peak_idx2 = int(peak_idxes2[k])
                    series_list = []
                    for df_base, freq in zip(base_df_list, measurement_freqs):
                        df_detrend = df_base.loc[:, [f'Detrend{column1}', f'Detrend{column2}']]
                        df_smoothed = df_base.loc[:, [f'Smoothed{column1}', f'Smoothed{column2}']]

                        # セグメント全体を取り出して、ピークが中央（TARGET_LEN//2）にくるように配置する
                        series_tmp1 = _extract_centered_segment(df_detrend.iloc[:, 0], start_idx1, end_idx1, peak_idx1, TARGET_LEN)
                        series_tmp2 = _extract_centered_segment(df_detrend.iloc[:, 1], start_idx2, end_idx2, peak_idx2, TARGET_LEN)
                        
                        # ベースライン補正（スムーズも同じ配置で切り出す）
                        if settings.baseline_correct_flag:
                            series_smoothed_tmp1 = _extract_centered_segment(df_smoothed.iloc[:, 0], start_idx1, end_idx1, peak_idx1, TARGET_LEN)
                            series_smoothed_tmp2 = _extract_centered_segment(df_smoothed.iloc[:, 1], start_idx2, end_idx2, peak_idx2, TARGET_LEN)
                            series_smoothed_tmp1 = series_smoothed_tmp1.replace(0, np.nan).fillna(1)
                            series_smoothed_tmp2 = series_smoothed_tmp2.replace(0, np.nan).fillna(1)
                            series_tmp1 = series_tmp1 / series_smoothed_tmp1
                            series_tmp2 = series_tmp2 / series_smoothed_tmp2

                        series_list.append(series_tmp1.rename(f'{series_tmp1.name}_{freq}'))
                        series_list.append(series_tmp2.rename(f'{series_tmp2.name}_{freq}'))

                        order1 = function.get_order(max(series_tmp1))
                        order1_list.append(order1)
                        order2 = function.get_order(max(series_tmp2))
                        order2_list.append(order2)
                    df_tmp = pd.concat(series_list, axis=1)
                    df_tmp = df_tmp.fillna(0) # 欠損値を0で補間
                    df_tmp.reset_index(inplace=True, drop=True)
                    # TARGET_LEN 固定なのでスキップ条件は不要（必要なら長さ超過時は continue する）
                    if len(df_tmp) != TARGET_LEN:
                        # 何らかの理由で長さが合わない場合は調整（通常ここは通らない）
                        df_tmp = df_tmp.reindex(range(TARGET_LEN)).fillna(0)
                    df_tmp_list.append(df_tmp)
            df_tmp_list_list.append(df_tmp_list)
        else:
            print('Unable to create dataset due to insufficient number of peaks.')
            with open(rf'{output_dir}\info.json', 'w', encoding="utf-8") as f:
                json.dump(settings_json, f, ensure_ascii=False, indent=4)
            sys.exit()

    order1 = statistics.mode(order1_list) if order1_list else None
    order2 = statistics.mode(order2_list) if order2_list else None
    
    # DataFrameを保存
    channel_num = len(measurement_freqs) * 2
    X, y = [], []
    label = 0

    for i, df_tmp_list in tqdm(enumerate(df_tmp_list_list), desc='output datasets', total=num_classes):
        if settings.save_peak_data_flag:
            dataset_dir = rf"{output_dirs[i]}\dataset"
            function.make_dirs(dataset_dir)
        if df_peak_list is None: # ピークが1つも検出されなかった場合はスキップ
            continue
        for j, df_tmp in enumerate(df_tmp_list):
            # 固定長なので穴埋め不要
            x = df_tmp.values
            X.append(x)
            y.append(label)

            # 抽出したピークデータをcsvに出力
            if settings.save_peak_data_flag:
                df_tmp.to_csv(rf"{dataset_dir}\{j}.csv", index=False)
        label += 1
    
    # データセットを出力
    np.save(rf"{output_dir}\X.npy", X)
    np.save(rf"{output_dir}\y.npy", y)
    settings_json['dataset_info'] = {
        'num_classes': num_classes,
        'data_length': TARGET_LEN,
        'baseline_correct_flag': settings.baseline_correct_flag,
        'measurement_freqs': measurement_freqs,
        'order1': order1,
        'order2': order2
    }
    with open(rf'{output_dir}\info.json', 'w', encoding="utf-8") as f:
        json.dump(settings_json, f, ensure_ascii=False, indent=4)
    print('Done')

def main():
    get_setting()
    peak_detect()
    make_time_series_data()

if __name__ == '__main__':
    main()
