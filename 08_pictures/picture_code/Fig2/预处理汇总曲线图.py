import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


COLOR_RAW = "#212121"
COLOR_SNV = "#1976D2"
COLOR_SG = "#8E24AA"
COLOR_STD = "#D32F2F"

LINEWIDTH = 1.5
FONT_FAMILY = "Arial"
FIG_SIZE = (7.5, 6)
FONT_SIZE_LABEL = 24
FONT_SIZE_TICK = 14
FONT_SIZE_LEGEND = 14

SG_WINDOW = 9
SG_POLY = 2
SG_DERIV = 1

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = [FONT_FAMILY]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

file_path = Path(__file__).resolve().parent / "original.xlsx"


def make_wavenumber_unit_superscript(svg_path):
    svg_path = Path(svg_path)
    text = svg_path.read_text(encoding="utf-8")
    text = text.replace(
        "Wavenumbers (cm-1)",
        'Wavenumbers (cm<tspan baseline-shift="super" font-size="70%">-1</tspan>)',
    )
    svg_path.write_text(text, encoding="utf-8")

try:
    df = pd.read_excel(file_path, usecols="FN:AGR", header=0)
    print(f"数据加载成功: {df.shape}")
    wavelengths = df.columns.astype(float)
    raw_spectrum = df.iloc[0].values
except Exception as e:
    print(f"读取文件出错，正在生成模拟数据... ({e})")
    wavelengths = np.linspace(400, 2500, 500)
    raw_spectrum = np.zeros_like(wavelengths)
    raw_spectrum += 1000 * np.exp(-((wavelengths - 700) ** 2) / (2 * 50 ** 2))
    raw_spectrum += 800 * np.exp(-((wavelengths - 1400) ** 2) / (2 * 80 ** 2))
    raw_spectrum += wavelengths * 0.2 + np.random.normal(0, 5, size=len(wavelengths))

y_raw = raw_spectrum
y_snv = (y_raw - np.mean(y_raw)) / np.std(y_raw)
y_sg = savgol_filter(y_snv, window_length=SG_WINDOW, polyorder=SG_POLY, deriv=SG_DERIV)

scaler = StandardScaler()
y_std = scaler.fit_transform(y_sg.reshape(-1, 1)).flatten()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True)

ax1.plot(wavelengths, y_raw, color=COLOR_RAW, linewidth=LINEWIDTH)
ax1.text(0.02, 0.85, "Step 1: Raw Data", transform=ax1.transAxes, fontsize=FONT_SIZE_LEGEND, color="black")

ax2.plot(wavelengths, y_snv, color=COLOR_SNV, linewidth=LINEWIDTH)
ax2.text(0.02, 0.85, "Step 2: SNV", transform=ax2.transAxes, fontsize=FONT_SIZE_LEGEND, color="black")

ax3.plot(wavelengths, y_sg, color=COLOR_SG, linewidth=LINEWIDTH)
ax3.text(0.02, 0.85, "Step 3: SG (1st Deriv)", transform=ax3.transAxes, fontsize=FONT_SIZE_LEGEND, color="black")

ax4.plot(wavelengths, y_std, color=COLOR_STD, linewidth=LINEWIDTH)
ax4.text(0.02, 0.85, "Step 4: Standardized", transform=ax4.transAxes, fontsize=FONT_SIZE_LEGEND, color="black")

ax4.set_xlabel("Wavenumbers (cm-1)", fontsize=FONT_SIZE_LABEL, labelpad=4)
fig.supylabel("Intensity", fontsize=FONT_SIZE_LABEL)

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_SIZE_TICK,
        direction="in",
        length=6,
        width=1.2,
        colors="black",
        pad=2,
        top=True,
        right=True,
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

plt.tight_layout()
plt.subplots_adjust(hspace=0.1)

file_name = "preprocessing_subplots.svg"
plt.savefig(file_name, format="svg", dpi=300)
make_wavenumber_unit_superscript(file_name)
print(f"图表已保存: {file_name}")

plt.show()
