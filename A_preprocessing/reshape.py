import pandas as pd
import os, re
import settings

path = settings.reshape_path

def reshape(input_path):
    ext = input_path.split(".")[-1]
    if ext == "txt" and re.search("[0-9]+k", input_path):
        print(f'{input_path} is being processed.')
        
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
                freq_list.pop(0)
        
        print('-'*100)
        return output_path

def main(path):
    if os.path.isdir(path):
        directories = os.listdir(path)
        for directory in directories:
            main(rf"{path}\{directory}")
    else:
        reshape(path)

if __name__ == "__main__":
    main(path)