import pandas as pd
import os
import settings

path_list = settings.merge_path_list

def main(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)

    df = dfs[0]
    time_span = [df.index.values[i] - df.index.values[i-1] for i in range(1, len(df))]
    time_span = sum(time_span) / len(time_span)
    for df_tmp in dfs[1:]:
        shift_time = time_span + df.index.values[-1]
        df_tmp.index += shift_time
        df = pd.concat([df, df_tmp])
    
    # ディレクトリとファイルの分割並びにリスト作成
    dir_list = []
    file_name_list = []
    for path in paths:    
        directory, file_name = os.path.split(path)
        dir_list.append(directory)
        file_name_list.append(file_name)
    
    #エラーの表示
    for file_name in file_name_list:
        if not 'detrend' in file_name:
            print('The data must be detrended before running this program.')
            return
    
    # ディレクトリの共通部分の抽出
    common_dir_name = os.path.commonpath(dir_list)

    # ファイルの共通部分の抽出
    common_file_names = []
    for file_name in file_name_list:
        common_file_name = file_name.split('_')[0] 
        common_file_names.append(common_file_name)

    if len(set(common_file_names)) == 1:
        pass
    else:
        print('Please make the first name of the file the measurement frequency')
        print('or')
        print('Set files with the same measurement frequency.')
        return
        
    # 新しいパスの出力
    output_path = rf"{common_dir_name}\{common_file_names[0]}_merge.csv"
    df.to_csv(output_path)
    print(f"{output_path} was created.")

if __name__ == "__main__":
    if isinstance(path_list[0], list):
        for paths in path_list:
            main(paths)
    else:
        main(path_list)