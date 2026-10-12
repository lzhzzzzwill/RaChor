import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


COLOR_ALL_SAMPLES = "#b1cbd4"
COLOR_STD_FILL = "#389cb7"
COLOR_MEAN = "#0000ff"

ALPHA_ALL_SAMPLES = 0.6
ALPHA_STD_FILL = 0.8
LINEWIDTH_MEAN = 1.2

FONT_FAMILY = "Arial"
FIG_SIZE = (7.5, 6)
FONT_SIZE_LABEL = 24
FONT_SIZE_TICK = 14
FONT_SIZE_LEGEND = 14

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
    df = pd.read_excel(file_path, usecols="B:BZF", header=0)
    print(f"数据加载成功: {df.shape}")
except Exception as e:
    print(f"读取文件出错，正在生成模拟数据... ({e})")
    wavelengths = np.linspace(400, 2500, 1000)
    n_samples = 187
    data = []
    for _ in range(n_samples):
        base = np.zeros_like(wavelengths)
        base += 1.0 * np.exp(-((wavelengths - 600) ** 2) / (2 * 50 ** 2))
        base += 0.8 * np.exp(-((wavelengths - 1200) ** 2) / (2 * 80 ** 2))
        base += 0.6 * np.exp(-((wavelengths - 1800) ** 2) / (2 * 60 ** 2))
        noise = np.random.normal(0, 0.05, size=len(wavelengths)) + np.random.rand() * 0.2
        data.append(base + noise)
    df = pd.DataFrame(data, columns=wavelengths)

x_axis = df.columns.astype(float)
mean_spectra = df.mean(axis=0)
std_spectra = df.std(axis=0)
min_spectra = df.min(axis=0)
max_spectra = df.max(axis=0)
upper_std = mean_spectra + std_spectra
lower_std = mean_spectra - std_spectra

fig, ax = plt.subplots(figsize=FIG_SIZE)

ax.fill_between(
    x_axis,
    min_spectra,
    max_spectra,
    color=COLOR_ALL_SAMPLES,
    alpha=ALPHA_ALL_SAMPLES,
    label="All samples",
    zorder=1,
)

ax.fill_between(
    x_axis,
    lower_std,
    upper_std,
    color=COLOR_STD_FILL,
    alpha=ALPHA_STD_FILL,
    label="Plus/minus 1 sigma",
    zorder=2,
)

ax.plot(
    x_axis,
    mean_spectra,
    color=COLOR_MEAN,
    linewidth=LINEWIDTH_MEAN,
    label="Mean",
    zorder=3,
)

ax.set_xlabel("Wavenumbers (cm-1)", fontsize=FONT_SIZE_LABEL, labelpad=4)
ax.set_ylabel("Intensity", fontsize=FONT_SIZE_LABEL, labelpad=6)
ax.legend(fontsize=FONT_SIZE_LEGEND, loc="upper right", frameon=False)

ax.set_xlim(x_axis.min(), x_axis.max())
ax.set_ylim(bottom=0)

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

for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color("black")

plt.tight_layout()

file_name = "spectra_overview.svg"
plt.savefig(file_name, format="svg", dpi=300)
make_wavenumber_unit_superscript(file_name)
print(f"图片已保存为: {file_name}")

plt.show()
