from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. 读取数据 ===
file_path = Path(__file__).resolve().parent / "original.xlsx"

# 读取 B 到 BZF 列
# header=0 表示第一行是波长(X轴坐标)
try:
    df = pd.read_excel(file_path, usecols="B:BZF", header=0)
    print(f"数据加载成功: {df.shape}")
except Exception as e:
    print(f"读取文件出错，正在生成模拟数据用于演示... ({e})")
    # --- 生成模拟数据 (如果找不到文件，这段代码保证你能运行看效果) ---
    wavelengths = np.linspace(400, 2500, 1000)  # 模拟波长
    n_samples = 187
    # 生成一些类似光谱的高斯波峰数据
    data = []
    for _ in range(n_samples):
        base = np.zeros_like(wavelengths)
        # 添加随机波峰
        base += 1.0 * np.exp(-((wavelengths - 600) ** 2) / (2 * 50 ** 2))
        base += 0.8 * np.exp(-((wavelengths - 1200) ** 2) / (2 * 80 ** 2))
        base += 0.6 * np.exp(-((wavelengths - 1800) ** 2) / (2 * 60 ** 2))
        # 添加随机噪声
        noise = np.random.normal(0, 0.05, size=len(wavelengths)) + np.random.rand() * 0.2
        data.append(base + noise)

    df = pd.DataFrame(data, columns=wavelengths)
    # -----------------------------------------------------------

# === 2. 数据处理 ===

# 获取 X 轴数据 (波长/光谱坐标)
# 假设列名就是波长数值
x_axis = df.columns.astype(float)

# 计算统计量
mean_spectra = df.mean(axis=0)  # 均值
std_spectra = df.std(axis=0)  # 标准差

# 计算上下限 (均值 ± 1倍标准差)
upper_bound = mean_spectra + std_spectra
lower_bound = mean_spectra - std_spectra

# === 3. 绘图 ===

# 设置画布大小 (宽:高)
fig, ax = plt.subplots(figsize=(10, 6))

# (A) 绘制所有样本 (Thin light-gray lines)
# 为了避免图例里出现187个"All samples"，我们只给第一条线加标签
ax.plot(x_axis, df.iloc[0], color='lightgray', linewidth=0.5, alpha=0.6, label='All samples')
# 绘制剩余的线
# 使用 values.T 可以一次性画出所有线，性能更好，但为了逻辑清晰，这里演示通常做法
ax.plot(x_axis, df.values.T, color='lightgray', linewidth=0.5, alpha=0.6)

# (B) 绘制 ±1σ 阴影带 (Semi-transparent band)
ax.fill_between(x_axis, lower_bound, upper_bound,
                color='gray', alpha=0.4, label='$\pm 1\sigma$')

# (C) 绘制均值线 (Thick dark mean spectrum)
# zorder=3 确保均值线在最上层
ax.plot(x_axis, mean_spectra, color='#404040', linewidth=2.5, label='Mean', zorder=3)

# === 4. 样式设置 ===

ax.set_title("Spectra Overview", fontsize=16, pad=15)
ax.set_xlabel("Spectral coordinate", fontsize=14)
ax.set_ylabel("Intensity", fontsize=14)

# 设置图例
# loc='best' 让系统自动寻找遮挡最少的位置
ax.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='gray')

# 调整坐标轴范围 (紧凑显示)
ax.set_xlim(x_axis.min(), x_axis.max())
ax.set_ylim(bottom=0)  # 假设强度不为负

# 如果你想要之前的 Times New Roman 字体，可以取消下面注释
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

plt.tight_layout()
plt.show()