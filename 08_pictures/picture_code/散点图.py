from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import os

# ==========================================
# === 1. 全局配置 (配置区域) ===
# ==========================================

# --- 1.1 核心颜色配置 (在这里修改四张图的颜色) ---
# 建议使用 Hex 颜色代码，也可以用 'red', 'blue' 等英文
COLOR_1_SN_GLOBAL = '#2e58ff'  # 图1颜色 (深蓝)
COLOR_2_SN_MOE = '#ff7f0e'  # 图2颜色 (橙色)
COLOR_3_BI_GLOBAL = '#2ca02c'  # 图3颜色 (绿色)
COLOR_4_BI_MOE = '#9467bd'  # 图4颜色 (紫色)

# --- 1.2 线条与通用配置 ---
FIT_LINE_COLOR = '#D62728'  # 拟合线颜色 (红色)
IDENTITY_LINE_COLOR = 'black'  # Y=T 线颜色 (黑色)

# --- 1.3 字体与画布 ---
FONT_FAMILY = 'Arial'
FIG_SIZE = (8, 8)
FONT_SIZE_LABEL = 34
FONT_SIZE_TICK = 28
FONT_SIZE_LEGEND = 26
FONT_SIZE_TEXT = 26

# --- 设置全局字体 ---
try:
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.sans-serif'] = [FONT_FAMILY]
except:
    pass
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# === 2. 核心绘图函数 (立体堆叠效果) ===
# ==========================================

def plot_3d_scatter_refined(ax, x, y, color, label):
    """
    实现“贪吃蛇”式的层层堆叠效果：
    后画的球体（主体+高光）会完全遮挡住先画的球体。
    """
    # 计算高光的偏移量
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    offset_x = (x_range if x_range > 0 else 1) * 0.006
    offset_y = (y_range if y_range > 0 else 1) * 0.006

    # --- 关键：循环绘制，确保层级遮挡 ---
    # base_zorder 设为 10，保证在网格线和拟合线之上
    base_zorder = 10

    # 遍历每一个点进行绘制
    for i, (xi, yi) in enumerate(zip(x, y)):
        # 1. 图例处理：只给第一个点加 label
        lbl = label if i == 0 else None

        # 2. 当前球体的层级 (随索引递增)
        current_z = base_zorder + i

        # 3. 绘制主体 (带颜色的球)
        ax.scatter(xi, yi, c=color, s=110, edgecolors='black', linewidth=0.6,
                   alpha=1.0, label=lbl, zorder=current_z)

        # 4. 绘制高光 (白色点)
        # 高光在当前主体之上 (+0.1)，但在下一个球体 (current_z + 1) 之下
        ax.scatter(xi - offset_x, yi + offset_y, c='white', s=20,
                   alpha=0.4, edgecolors='none', zorder=current_z + 0.1)


# ==========================================
# === 3. 数据读取 (带自动模拟数据) ===
# ==========================================

file_name = Path(__file__).resolve().parent / "nestedcv_oof_predictions.xlsx"


def generate_mock_data(n=100, noise=0.5):
    true_val = np.random.uniform(5, 25, n)
    true_val = np.sort(true_val)  # 排序以产生线条感
    pred_val = true_val + np.random.normal(0, noise, n)
    return pd.DataFrame({'true': true_val, 'pred': pred_val})


try:
    if os.path.exists(file_name):
        print(f"正在读取文件: {file_name}")
        df_global = pd.read_excel(file_name, sheet_name="GlobalStack")
        df_moe = pd.read_excel(file_name, sheet_name="MoE")
    else:
        raise FileNotFoundError("文件不存在")
except Exception as e:
    print(f"提示: 未找到数据文件，正在生成【模拟数据】演示效果...")
    df_global = generate_mock_data(150, 0.8)
    df_global.columns = ['118Sn (KED)_true', '118Sn (KED)_pred']

    df_moe = generate_mock_data(150, 0.5)
    df_moe.columns = ['118Sn (KED)_true', '118Sn (KED)_pred']

    # 模拟 Bi 数据
    df_global['209Bi (KED)_true'] = df_global['118Sn (KED)_true'] + 10
    df_global['209Bi (KED)_pred'] = df_global['118Sn (KED)_pred'] + 10
    df_moe['209Bi (KED)_true'] = df_moe['118Sn (KED)_true'] + 10
    df_moe['209Bi (KED)_pred'] = df_moe['118Sn (KED)_pred'] + 10


# ==========================================
# === 4. 绘图主逻辑 ===
# ==========================================

def plot_scatter(df, true_col, pred_col, filename, point_color):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    y_true = df[true_col].values
    y_pred = df[pred_col].values

    # --- A. 指标计算 ---
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / rmse if rmse != 0 else 0
    stats_text = f"R² = {r2:.5f}\nRMSE = {rmse:.5f}\nRPD = {rpd:.5f}"

    # --- B. 坐标范围与回归线 ---
    data_min = min(y_true.min(), y_pred.min())
    data_max = max(y_true.max(), y_pred.max())
    padding = (data_max - data_min) * 0.1
    limit_min, limit_max = data_min - padding, data_max + padding

    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    line_x = np.array([limit_min, limit_max])

    # --- C. 辅助线绘制 ---
    ax.plot([limit_min, limit_max], [limit_min, limit_max],
            color=IDENTITY_LINE_COLOR, linestyle=(0, (5, 5)), linewidth=1.2, zorder=1, label='Y=T')

    ax.plot(line_x, slope * line_x + intercept,
            color=FIT_LINE_COLOR, linestyle='--', linewidth=2.0, zorder=2, label='Fit')

    # --- D. 绘制立体散点 (传入指定的颜色) ---
    plot_3d_scatter_refined(ax, y_true, y_pred, point_color, 'Data')

    # --- E. 样式与边框 ---
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK,
                   direction='in', length=8, width=1.0,
                   top=False, right=False)

    ax.set_xlabel("Measured Value", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Predicted Value", fontsize=FONT_SIZE_LABEL)

    # --- F. 指标文本 ---
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=FONT_SIZE_TEXT, verticalalignment='top', linespacing=1.6)

    # --- G. 图例排序 ---
    handles, labels = ax.get_legend_handles_labels()
    order_map = {lbl: hdl for hdl, lbl in zip(handles, labels)}
    sorted_handles = []
    sorted_labels = []

    desired_order = ['Data', 'Fit', 'Y=T']
    for name in desired_order:
        if name in order_map:
            sorted_handles.append(order_map[name])
            sorted_labels.append(name)

    ax.legend(sorted_handles, sorted_labels,
              loc='lower right', fontsize=FONT_SIZE_LEGEND, frameon=False)

    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300)
    print(f"已生成: {filename} (颜色: {point_color})")
    plt.show()


# ==========================================
# === 5. 循环执行任务 (配置颜色) ===
# ==========================================

# 格式: (DataFrame, 真实值列名, 预测值列名, 文件名, 颜色变量)
tasks = [
    (df_global, '118Sn (KED)_true', '118Sn (KED)_pred', 'Scatter_Sn_Global.svg', COLOR_1_SN_GLOBAL),
    (df_moe, '118Sn (KED)_true', '118Sn (KED)_pred', 'Scatter_Sn_MoE.svg', COLOR_2_SN_MOE),
    (df_global, '209Bi (KED)_true', '209Bi (KED)_pred', 'Scatter_Bi_Global.svg', COLOR_3_BI_GLOBAL),
    (df_moe, '209Bi (KED)_true', '209Bi (KED)_pred', 'Scatter_Bi_MoE.svg', COLOR_4_BI_MOE)
]

for df, t_col, p_col, fname, color in tasks:
    plot_scatter(df, t_col, p_col, fname, point_color=color)
