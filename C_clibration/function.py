import numpy as np
import os, shutil

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

def remove_outer(num_array: list[float]) -> list[float]:
    q75, q25 = np.percentile(num_array, [75 ,25])
    iqr = q75 - q25
    num_max = q75 + iqr * 1.5
    num_min = q25 - iqr * 1.5
    num_array = num_array[num_min < num_array]
    num_array = num_array[num_array < num_max]
    return num_array

def get_diameter_list(cal_list: list[float], sample_list: list[float], d: float) -> list[float]:
    #外れ値除去
    cal_list = remove_outer(cal_list)
    sample_list = remove_outer(sample_list)
    
    mean_cal = sum(cal_list) / len(cal_list)
    diameter_list = ((sample_list / mean_cal) ** (1 / 3)) * d
    return diameter_list

# 直径から平均や分散などを計算
def get_info(diameters: list[float]) -> list[float]:
    info_keys = ["Mean diameter", "Median diameter", "CV", "Peak number"]
    mean = np.mean(diameters)
    median = np.median(diameters)
    std = np.std(diameters)
    cv = std / mean
    num = len(diameters)
    info_values =  [mean, median, cv, num]
    dic = dict(zip(info_keys, info_values))
    return dic