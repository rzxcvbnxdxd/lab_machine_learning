

# reshape

#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng4\5k_deng.txt"
#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA\FluA_5k.txt"
#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA_AICHI\FluAAICHI_5k.txt"
#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA_UDORN\FluUDON_5k.txt"
#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\FluB\Deng3_5k.txt"
#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng3\Deng3_5k.txt"
#reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng2\Deng2_5k.txt"
reshape_path = r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng1\Deng1_5k.txt"

# detrend
detrend_path = [

r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng4\deng4_5k.csv",
r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng3\5k_01.csv",
r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng2\5k.csv",
r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng1\5k.csv",
r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA\5k.csv",
r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA_AICHI\5k.csv",
r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA_UDORN\5k.csv",



]


'''
<detrend_pathの指定方法について>
単一ののcsvファイルをトレンド補正する場合は


detrend_path = r'hoge/~~k.csv'


のように記述する。
複数のcsvファイルをトレンド補正する場合は


detrend_path = [r'hoge/~~k.csv',
                r'hoge/~~k.csv',
                r'hoge/~~k.csv',
                r'hoge/~~k.csv']


のようにリストで記述する。
'''


# データにカットしたい部分がある場合は以下を記述（cutの必要のない場合は空のリストを指定）
start_times = [] # 最初からの場合 "0"
end_times = [] # での場合 "-1"


# merge
merge_path_list = [
[

],
# [
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4700k_01_detrend.csv",
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4700k_02_detrend.csv",
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4700k_03_detrend.csv",
 
# ],


# [
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4800k_01_detrend.csv",
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4800k_02_detrend.csv",
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4800k_03_detrend.csv",
 
# ],
# [
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4900k_01_detrend.csv",
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4900k_02_detrend.csv",
# r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241224_同軸インピーダンス測定\AΦ100-1\やり直し\4900k_03_detrend.csv",
 
# ],


]
'''
<merge_path_listの指定方法について>
'hoge1/~~k_01.csv',と'hoge1/~~k_02.csv'を結合(merge)させたい場合は


merge_path_ist = [r'hoge1/~~k_01.csv', r'hoge1/~~k_02.csv']


と指定する。
上記に加えて'hoge2/~~k_01.csv',と'hoge2/~~k_02.csv'を結合させたい場合は


merge_path_ist = [
                     [
                         r'hoge1/~~k_01.csv',
                         r'hoge1/~~k_02.csv'
                     ],
                     [
                         r'hoge2/~~k_01.csv',
                         r'hoge2/~~k_02.csv',
                     ]
                 ]
と、リストをネストさせる形で指定する。       
'''
