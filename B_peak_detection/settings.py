samples = {# "sample_name": r"E:\data\~~~",
"Deng4": r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng4\deng4_5k_detrend.csv",
"Deng3": r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng3\5k_01_detrend.csv",
"Deng2": r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng2\5k_detrend.csv",
"Deng1": r"C:\Users\rzxcv\Downloads\Zurich_Red\Deng1\5k_detrend.csv",
"FluA": r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA\5k_detrend.csv",
"FluA_AICHI": r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA_AICHI\5k_detrend.csv",
"FluA_UDORN": r"C:\Users\rzxcv\Downloads\Zurich_Red\FluA_UDORN\5k_detrend.csv",
# "1790": r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241029_乳酸菌（寒天）\9本\1790\1000k_detrend.csv",
# "1995": r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241029_乳酸菌（寒天）\9本\1995\1000k_detrend.csv",
# "3324": r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241029_乳酸菌（寒天）\9本\3324\1000k_detrend.csv",
# "4140": r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241029_乳酸菌（寒天）\9本\4140\1000k_detrend.csv",
#"1400k":r"C:\Users\riku3\東京工業大学・山本研 Dropbox\ラボ共有(チーム)\佐藤（璃）\実験データ\20241222_インピーダンス測定\W9H15L18_main\3μm\COOH\1400k_detrend.csv",


}
# Directory to save analysis results
output_path = None
#r"C:\Users\Ken_Hayashida\東京工業大学・山本研 Dropbox\交流ナノポア\実験データ\測定データ\同軸リングフィルム\20240328\analysis_file\20k\お試し\コード変更前\ウイルス全種" # {None, any directory path}

# Do you want to create a data set from multiple frequencies?
 # {None, list of other frequency values}
multi_freqs = None # 必要なら周波数のリストを指
# correct by baseline value?

baseline_correct_flag = True # {True or False}

# Output peak data?
save_peak_data_flag = False # {True or False}

# find peaks
threshold_mode = "auto" # {"auto", "manual"}
pdf = 4.0  # Required if you selected "auto"
thresholds = None # Required if you selected "manual"
find_peaks_mode = "normal" # {"scipy", "normal"}

'''
Do not change the below
'''
params = {
	'samples': samples,
	'find_peaks':{
            'threshold_mode': threshold_mode,
		'pdf': pdf,
            'thresholds': thresholds,
		'find_peaks_mode': find_peaks_mode
	}
}