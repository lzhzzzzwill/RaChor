from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # 引入此模块用于自定义图例

# ==========================================
# === 1. 全局配置 (Configuration) ===
# ==========================================

# --- 字体与画布设置 ---
FONT_FAMILY = 'Arial'
FIG_SIZE = (10, 6)
FONT_SIZE_LABEL = 30  # 坐标轴标题字号
FONT_SIZE_TICK = 26  # 刻度字号
FONT_SIZE_LEGEND = 20  # 图例字号

# --- 颜色设置 (5个Fold的鲜艳颜色) ---
FOLD_COLORS = ['#007BFF', '#FF0000', '#00C853', '#AA00FF', '#FFAB00']
FOLD_LABELS = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']

# --- 箱线图与散点样式 ---
BOX_WIDTH = 0.6
BOX_LINEWIDTH = 1.5
MEDIAN_COLOR = 'black'

# --- 散点水平偏移量 ---
X_OFFSETS = [-0.14, -0.07, 0.0, 0.07, 0.14]

# ==========================================
# === 2. 数据准备 (Train OOF 数据) ===
# ==========================================

import numpy as np


# 118Sn (KED)
DATA_SN = {
    "PLS": [0.926941647, 0.917607362, 0.918796202, 0.942638888, 0.938199953],
    "KRR": [0.941771467, 0.924947575, 0.935650362, 0.977086242, 0.935981091],
    "ElasticNet": [0.930662580, 0.918342368, 0.920481659, 0.931475024, 0.926104435],
    "RFRR": [0.931230918, 0.925926675, 0.939050521, 0.953324854, 0.912890429],
    "RSRidge": [0.932519863, 0.921482790, 0.921576402, 0.943893900, 0.928192527],

    "LFR stack": [0.8479, 0.9166, np.nan, 0.8179, np.nan],
    "MFR stack": [0.9131, 0.9015, 0.8989, 0.9298, 0.9135],
    "HFR stack": [0.8426, np.nan, 0.8712, np.nan, 0.8228],

    "Global Stack": [0.943892942, 0.926353946, 0.931495845, 0.963758578, 0.943574087],
    "Final Fit": [0.945474076, 0.926248669, 0.924984479, 0.954963489, 0.939114023]
}


# 209Bi (KED)
DATA_BI = {
    "PLS": [0.896580538, 0.881017036, 0.884609961, 0.892643571, 0.910689983],
    "KRR": [0.901091427, 0.883671213, 0.902421467, 0.936774297, 0.913931831],
    "ElasticNet": [0.898302656, 0.880301898, 0.887867438, 0.894691945, 0.911660094],
    "RFRR": [0.906763185, 0.886284368, 0.920446583, 0.909066515, 0.911951966],
    "RSRidge": [0.898590319, 0.881634934, 0.883416814, 0.870879074, 0.911719358],

    "LFR stack": [0.8255, 0.8727, np.nan, 0.8345, 0.8461],
    "MFR stack": [0.8607, 0.8660, 0.8643, 0.8488, 0.8779],
    "HFR stack": [0.8511, np.nan, 0.8148, np.nan, 0.8677],

    "Global Stack": [0.904187761, 0.880953606, 0.897989758, 0.916404649, 0.911440966],
    "Final Fit": [0.908245297, 0.877473006, 0.887504479, 0.911498381, 0.921610497]
}


# 测试集 R²：04_Supplementary_Data.xlsx / Table S4，按 Fold 1–5 排列。

DATA_SN_TEST = {
    'PLS': [0.907897063994715, 0.844947526839587, 0.940960084412616, 0.865262008141208, 0.840914237246731],
    'KRR': [0.934484682839108, 0.865028979205522, 0.95251027019325, 0.875286499064924, 0.821875842338303],
    'ElasticNet': [0.928423740247887, 0.85275311719939, 0.946184309544151, 0.879626586317098, 0.848350727294251],
    'RFRR': [0.908559036889747, 0.868344586117299, 0.953886922526651, 0.884725527719118, 0.790224544915325],
    'RSRidge': [0.931505425646202, 0.85917590381301, 0.950088312243597, 0.872701463463512, 0.853306155457592],
    'Global Stack': [0.943885716015629, 0.871016641650489, 0.959583108111501, 0.89508613361033, 0.902154065160273],
    'Final Fit': [0.943905632412836, 0.868375087775397, 0.959247318804589, 0.896734293435833, 0.907725661803159],
}

DATA_BI_TEST = {
    'PLS': [0.880835686645516, 0.717716047354585, 0.899255739828442, 0.856952273419001, 0.720211836150321],
    'KRR': [0.8938445135743, 0.729130974104792, 0.906967092868551, 0.866196394546775, 0.757924081586224],
    'ElasticNet': [0.896828410671133, 0.713529715885763, 0.904777491413676, 0.860833905095057, 0.742148139770668],
    'RFRR': [0.900222265309872, 0.736070004430657, 0.909369305061259, 0.87262515404212, 0.796224276294488],
    'RSRidge': [0.879743614822951, 0.721766424691199, 0.910760024460926, 0.868359245248989, 0.743198017412493],
    'Global Stack': [0.898303844014295, 0.725927445338324, 0.92093544521713, 0.882854267998269, 0.821412760150141],
    'Final Fit': [0.904635489568976, 0.720982772757148, 0.922701753958771, 0.885631115712392, 0.844083063803324],
}


# ==========================================
# === 3. 绘图函数 ===
# ==========================================

def plot_performance(data_dict, filename_suffix, dataset="train"):
    # 设置全局字体
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.sans-serif'] = [FONT_FAMILY]
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    # 提取数据
    labels = list(data_dict.keys())
    data_values = list(data_dict.values())

    # 准备箱线图数据 (过滤NaN)
    boxplot_data = []
    for d in data_values:
        clean_d = [x for x in d if not np.isnan(x)]
        boxplot_data.append(clean_d)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 1. 绘制箱线图
    ax.boxplot(boxplot_data,
               tick_labels=labels,
               widths=BOX_WIDTH,
               patch_artist=False,
               showfliers=False,
               medianprops={'color': MEDIAN_COLOR, 'linewidth': BOX_LINEWIDTH},
               boxprops={'linewidth': BOX_LINEWIDTH},
               whiskerprops={'linewidth': BOX_LINEWIDTH},
               capprops={'linewidth': BOX_LINEWIDTH})

    # 2. 绘制散点 (Overlaid Scatter with Offsets)
    for i, model_scores in enumerate(data_values):
        x_base = i + 1
        for fold_idx, score in enumerate(model_scores):
            if np.isnan(score):
                continue

            # 水平偏移
            x_pos = x_base + X_OFFSETS[fold_idx]
            color = FOLD_COLORS[fold_idx]

            # 绘制点
            ax.scatter(x_pos, score, color=color, s=60, edgecolors='black', linewidth=0.8, zorder=3)

    # 3. 样式调整
    ylabel = "R² (Train CV)" if dataset == "train" else "R² (Test)"
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)

    # 统一 Y 轴尺度
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))

    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=FONT_SIZE_TICK)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
    ax.tick_params(direction='in', length=6, width=1)

    # 4. 图例 (修改点：frameon=False 去掉边框)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=FOLD_LABELS[i],
                              markerfacecolor=FOLD_COLORS[i], markersize=9, markeredgecolor='black')
                       for i in range(5)]

    # frameon=False: 移除图例的边框和背景
    if dataset == 'train':
        ax.legend(handles=legend_elements, loc='lower left', fontsize=FONT_SIZE_LEGEND, frameon=False)

    plt.tight_layout()

    # 保存
    prefix = "train_oof_no_frame" if dataset == "train" else "test_uniform_scale"
    output_filename = Path(__file__).resolve().parent / f"model_performance_{prefix}_{filename_suffix}.svg"
    plt.savefig(output_filename, format='svg', dpi=300)
    print(f"图表已保存: {output_filename}")
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


# ==========================================
# === 4. 执行绘图 ===
# ==========================================

if __name__ == "__main__":
    print("正在绘制 Sn 训练集 ...")
    plot_performance(DATA_SN, "118Sn", "train")
    print("正在绘制 Bi 训练集 ...")
    plot_performance(DATA_BI, "209Bi", "train")
    print("正在绘制 Sn 测试集 ...")
    plot_performance(DATA_SN_TEST, "118Sn", "test")
    print("正在绘制 Bi 测试集 ...")
    plot_performance(DATA_BI_TEST, "209Bi", "test")
