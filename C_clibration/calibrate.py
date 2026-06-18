import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
import matplotlib.ticker as tick
from datetime import datetime
import json
from scipy.stats import norm

import settings, function

cal_color = '#858f83'
sample_color = '#ad8585'
colors = ["#B89C9C", "#cca379", "#e4dab3", "#afb374", "#c0d6d6", "#708090", "#b3aeba", "#b3aeba"]

def make_txt(info: dict):
    txt = [rf'Mean Diameter:${round(info["Mean diameter"], 2)}\:nm$',
           rf'Median Diameter:${round(info["Median diameter"], 2)}\:nm$',
           rf'CV:${round(info["CV"], 3)}$',
           rf'Peak Number:${info["Peak number"]}$',
           rf'Particle Rate:${round(info["Particle rate"], 2)}\:/min$']
    txt = '\n'.join(txt)
    return txt

def make_histgram_one(cal_dia, sample_dia, cal_info, sample_info):
    fig = plt.figure(figsize=(19.20, 10.80))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Diameter[nm]', fontsize=24)
    ax.set_title(f'Particle Size Distribution of {sample_info["Name"]}', fontsize=30, pad=32)
    ax.set_ylabel('Count', fontsize=24)
    plt.subplots_adjust(left=0.1, right=0.76)
    ax_pos = ax.get_position()
    
    # measurement condition
    measurement_condition = ["Measurement Condition",
                             rf"Pressure: {settings.pressure}",
                             rf"Applied Voltage: ${settings.input_voltage}\:mV$",
                             rf"LPF: ${settings.lpf}\:$, BW: ${settings.bw}\:$"]
    measurement_condition = '\n'.join(measurement_condition)
    ax.text(ax_pos.x1 + 0.28, 1.06, measurement_condition, va='bottom', ha='left', transform=ax.transAxes, fontsize=12)
    
    # histgram
    for particle_name, diameter, color in zip([cal_info["Name"], sample_info["Name"]], [cal_dia, sample_dia], [cal_color, sample_color]):
        max_diameter = max(diameter)
        min_diameter = min(diameter)
        bins = np.arange(min_diameter, max_diameter + settings.binwidth, settings.binwidth)
        ax.hist(diameter, bins=bins, edgecolor='None', alpha=0.5, color=color, label=particle_name)
        
        # make norm
        if settings.norm_flag:
            x = np.linspace(max_diameter, min_diameter, 100)
            y = norm.pdf(x, np.mean(diameter), np.std(diameter)) * settings.binwidth * len(diameter)
            ax.plot(x, y, color=color, label=f'{particle_name} norm')
    
    # set grid
    ax.grid(which='major', linestyle='solid', color='gray', alpha=1)
    ax.grid(which='minor', linestyle="dotted", color='gray', alpha=0.8)

    # info
    ax.text(ax_pos.x1 + 0.28, 0.95, rf'Sample Particle : {sample_info["Name"]}', va='top', ha='left', transform=ax.transAxes, fontweight = 'bold', fontsize=16)
    ax.text(ax_pos.x1 + 0.28, 0.90, make_txt(sample_info), va='top', ha='left', transform=ax.transAxes, fontsize=12, linespacing=2)
    ax.text(ax_pos.x1 + 0.28, 0.45, rf'Calibration Particle : {cal_info["Name"]}', va='top', ha='left', transform=ax.transAxes, fontweight = 'bold', fontsize=16)
    ax.text(ax_pos.x1 + 0.28, 0.40, make_txt(cal_info), va='top', ha='left', transform=ax.transAxes, fontsize=12, linespacing=2)
    
    plt.gca().get_yaxis().set_major_locator(tick.MaxNLocator(integer=True)) # y軸の目盛りを整数だけにする
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if settings.x_limit:
        plt.xlim(*settings.x_limit)
    if settings.y_limit:
        plt.ylim(*settings.y_limit)

    plt.legend(fontsize=12, loc='upper right')

def make_histgram_all(df, names):
    sample_num = len(names)
    fig, ax = plt.subplots(sample_num, 1, figsize=(12,8), sharex='all')
    plt.subplots_adjust(hspace=0)
    plt.xlabel('Diameter[nm]', fontsize=30)
    fig.text(0.06, 0.5, 'Ratio [%]', fontsize=20, ha='center', va='center', rotation='vertical')
    
    # カラーマップの作成
    # カラーマップのセグメント位置
    segment_positions = np.linspace(0, 1, len(colors))

    # カラーマップのセグメント
    segments = {'red': [], 'green': [], 'blue': []}

    # 各色をRGBに分解してセグメントに追加
    for i, color in enumerate(colors):
        r, g, b = tuple(int(color.lstrip('#')[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        position = segment_positions[i]
        for key, value in zip(['red', 'green', 'blue'], [r, g, b]):
            segments[key].append((position, value, value))
    cmap = LinearSegmentedColormap("cmap", segments)
    
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))

    # plt.figure(figsize=(8, 2))
    # plt.imshow(gradient, aspect='auto', cmap=cmap)
    # plt.axis('off')
    
    max_diameters, min_diameters = [], []
    for i, name in enumerate(names):
        diameter = df[name].to_list()
        max_diameter = max(diameter)
        min_diameter = min(diameter)
        max_diameters.append(max_diameter)
        min_diameters.append(min_diameter)
        bins = np.arange(min_diameter, max_diameter + settings.binwidth, settings.binwidth)
        counts, _ = np.histogram(diameter, bins=bins)
        counts_normed = list(counts * 100 / sum(counts))
        ax[i].hist(bins[1:], bins=bins, facecolor=cmap(i/sample_num), edgecolor='black', alpha=1, label=name, weights=counts_normed)
    
    # グラフの範囲
    max_diameter = max(max_diameters)
    min_diameter = min(min_diameters)
    dia_range = max_diameter - min_diameter
    x_min = min_diameter - dia_range * 0.1
    x_max = max_diameter + dia_range * 0.1
    for i in range(len(names)):
        ax[i].set_xlim([x_min, x_max])
    
    lines = []
    labels = []
    for axis in fig.axes:
        axLine, axLabel = axis.get_legend_handles_labels()
        lines += axLine
        labels += axLabel
    plt.xticks(fontsize=26)
    fig.legend(lines, labels, bbox_to_anchor=(0.75, 0.87), loc='upper left', borderaxespad=0, fontsize=16)

def main():
    output_dir = rf"{settings.path}\calibration_histgram\{datetime.now().strftime('%y%m%d_%H%M%S')}"
    function.make_dirs(output_dir)
    
    with open(rf'{settings.path}\info.json', 'r', encoding='utf-8') as f:
        info_json = json.load(f)
    
    calibration_flag = False
    if settings.cal_particle is not None:
        calibration_flag = True
        cal = pd.read_csv(rf"{settings.path}\{settings.cal_particle}\peak.csv", index_col=0)
        cal_heights = cal['Amplitude(A)_PeakValue'].values
        cal_dias = function.get_diameter_list(cal_heights, cal_heights, settings.cal_diameter)
        cal_info = function.get_info(cal_dias)
        cal_info["Name"] = settings.cal_particle
        cal_info["Particle rate"] = cal_info["Peak number"]/info_json["samples"][settings.cal_particle]["measurement_time"]
    else:
        cal_dias = None
    
    if settings.sample_particles == "all":
        sample_particles = info_json["samples"].keys()
        if settings.cal_particle is not None:
            sample_particles = [particle for particle in sample_particles if particle!=settings.cal_particle]
    else:
        sample_particles = settings.sample_particles
    df = pd.DataFrame()
    for sample_particle in sample_particles:
        sample = pd.read_csv(rf"{settings.path}\{sample_particle}\peak.csv", index_col=0)
        sample_heights = sample['Amplitude(A)_PeakValue'].values
        if calibration_flag:
            sample_dias = function.get_diameter_list(cal_heights, sample_heights, settings.cal_diameter)
        else:
            sample_dias = sample_heights
        sample_info = function.get_info(sample_dias)
        sample_info["Name"] = sample_particle
        sample_info["Particle rate"] = sample_info["Peak number"]/info_json["samples"][sample_particle]["measurement_time"]
        if calibration_flag:
            make_histgram_one(cal_dias, sample_dias, cal_info, sample_info)
        df = df.join(pd.DataFrame({sample_particle: sample_dias}), how='outer')

        plt.savefig(rf"{output_dir}\{sample_particle}.svg")
        plt.close()
    
    make_histgram_all(df, sample_particles)
    plt.savefig(rf"{output_dir}\vertical_histgram.svg")
    plt.show()
    plt.close()
    
    df.to_csv(rf"{output_dir}\diameters.csv", index=False)

if __name__ == "__main__":
    main()
