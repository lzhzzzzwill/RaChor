from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 0. 定义需要高亮的特征波长
# ==========================================
selected_waves = [
    400.27, 427.67, 429.95, 448.13, 484.30, 526.93, 540.33, 551.46, 665.85,
    745.73, 747.87, 826.47, 853.80, 901.78, 905.93, 908.01, 910.08, 912.15,
    916.30, 926.64, 973.93, 1059.12, 1079.19, 1158.63, 1160.60, 1275.37,
    1326.97, 1422.90, 1454.44, 1478.41, 1491.26, 1509.56, 1565.83, 1681.56,
    1695.59, 1716.56, 1737.43, 1754.74, 1790.88, 1799.44
]

# ==========================================
# 1. 数据读取与处理
# ==========================================
file_path = Path(__file__).resolve().parent / "filter_snv_sg_scaled.csv"

try:
    # 读取 CSV (header=0 表示第一行是列名，即波长)
    df = pd.read_csv(file_path, header=0)

    # 获取光谱数据 (Y轴): 第1行(索引1)，第1列之后
    y_raw = df.iloc[1, 1:708].to_numpy(dtype=float)

    # 获取波长数据 (X轴): 表头第1列之后
    # 注意：表头读进来通常是字符串，需要转为 float
    x_raw = df.columns[1:708].astype(float).to_numpy()

except Exception as e:
    print(f"CSV读取失败，尝试Excel或其他方式: {e}")
    # 备用方案... (根据实际情况调整)
    exit()

# 处理 NaN
y_raw = np.nan_to_num(y_raw)

# 归一化 Y轴 (0-1)
scaler = MinMaxScaler()
y_final = scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

# ==========================================
# 2. 匹配选中波长的索引
# ==========================================
# 找出 selected_waves 在 x_raw 中对应的索引
highlight_indices = []
highlight_x = []
highlight_y = []

for target in selected_waves:
    # 找到与 target 差值最小的那个波长的索引
    idx = (np.abs(x_raw - target)).argmin()

    highlight_indices.append(idx)
    highlight_x.append(x_raw[idx])
    highlight_y.append(y_final[idx])

# ==========================================
# 3. 绘图 (Icon Style)
# ==========================================
fig, ax = plt.subplots(figsize=(4, 2.5))

# A. 绘制主要光谱曲线 (深灰色)
# 使用真实的波长 x_raw 作为横坐标
ax.plot(x_raw, y_final, color='#444444', linewidth=1.2, alpha=0.6, zorder=1)

# B. 高亮选取的波段 (红色圆点)
ax.scatter(highlight_x, highlight_y, color='#FF4500', s=15, zorder=2, label='Selected Bands')

# C. (可选) 添加垂直指示线，如果希望更明显可以取消注释下面这行
# ax.vlines(highlight_x, 0, highlight_y, color='#FF4500', linewidth=0.5, alpha=0.3, zorder=1)

# ==========================================
# 4. 样式调整 (保持原有 Icon 风格)
# ==========================================

# 隐藏右边和上边的框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置左边和下边的框样式
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 让轴贴紧数据范围
ax.set_xlim(x_raw.min(), x_raw.max())
ax.set_ylim(-0.05, 1.05)

# 移动脊柱位置
ax.spines['left'].set_position(('data', x_raw.min()))
ax.spines['bottom'].set_position(('data', -0.05))

# 隐藏刻度数字 (Icon风格通常不显示具体数字)
ax.set_xticks([])
ax.set_yticks([])

# 添加坐标轴末端箭头
# 注意：由于使用了真实坐标，箭头位置需要重新通过 transform 定位
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=6)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=6)

# 调整布局
plt.tight_layout()

# 保存
plt.savefig('preprocess_icon_highlight.svg', format='svg', transparent=True, dpi=300)
plt.show()

print(f"已高亮 {len(highlight_x)} 个特征波段。")