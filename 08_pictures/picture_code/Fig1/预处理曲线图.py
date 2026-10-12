from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 数据读取
# ==========================================
file_path = Path(__file__).resolve().parent / "filter_snv_sg_scaled.csv"

try:
    # 尝试读取 CSV
    df = pd.read_csv(file_path, header=0)
    # 取 B 列 (索引1) 到 AAF 列 (索引707左右)
    # 这里我们直接取第2列之后的所有数据作为光谱数据
    y_raw = df.iloc[1, 1:708].to_numpy(dtype=float)
except:
    # 如果实际上是 Excel 只是命名错了
    df = pd.read_excel(file_path, header=0)
    y_raw = df.iloc[1, 1:708].to_numpy(dtype=float)

# 处理可能存在的 NaN
y_raw = np.nan_to_num(y_raw)

# 归一化 (让曲线在 0-1 之间，适合画示意图)
scaler = MinMaxScaler()
y_final = scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()

# 构造 X 轴 (根据数据长度)
x = np.linspace(0, 100, len(y_final))

# ==========================================
# 2. 绘图 (Icon Style)
# ==========================================
fig, ax = plt.subplots(figsize=(4, 2.5)) # 宽长比接近原图

# 绘制光谱曲线 (深灰色)
ax.plot(x, y_final, color='#444444', linewidth=1.5)

# ==========================================
# 3. 添加右上角的文字标注 (复刻原图)
# ==========================================
# 原图有四行文字：N SNV, ~ SG, 此时 scale, outlier
# 我们用 text 和 unicode 符号来模拟

font_size = 9
text_color = '#444444'
x_text = 0.95  # 右对齐位置
y_text_start = 0.95
line_spacing = 0.12


# ==========================================
# 4. 样式调整 (坐标轴与边框)
# ==========================================

# 隐藏右边和上边的框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置左边和下边的框样式 (带箭头)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_position(('data', x.min())) # 轴贴紧数据
ax.spines['bottom'].set_position(('data', -0.05))

# 隐藏刻度数字
ax.set_xticks([])
ax.set_yticks([])

# 添加坐标轴末端箭头
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=6)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=6)

# 调整布局
plt.tight_layout()

# 保存
plt.savefig('preprocess_icon.svg', format='svg', transparent=True, dpi=300)
plt.show()