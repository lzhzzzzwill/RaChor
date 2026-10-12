from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

data_path = Path(__file__).resolve().parent / "filter_snv_sg_scaled_del_4_model.csv"
df = pd.read_csv(data_path)


def col_letter_to_index(col_letter):
    col_letter = col_letter.strip().upper()
    idx = 0
    for ch in col_letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


x_start = col_letter_to_index("B")
x_end = col_letter_to_index("AAF")
X = df.iloc[:, x_start : x_end + 1].values
target_metals = ["118Sn (KED)", "209Bi (KED)"]
y = df[target_metals].values

print(f"成功加载数据: {X.shape}")


def plot_3d_scatter(ax, x, y, color, label):
    ax.scatter(
        x,
        y,
        c=color,
        s=80,
        edgecolors="black",
        linewidth=0.5,
        alpha=1.0,
        label=label,
        zorder=2,
    )
    ax.scatter(x, y, c="white", s=25, alpha=0.4, edgecolors="none", zorder=3)


color_low = "#808000"
color_high = "#0000ff"

for i, col_name in enumerate(target_metals):
    data = y[:, i]
    samples = np.arange(1, len(data) + 1)

    mask_low = data <= 100
    mask_high = data > 100

    fig, ax = plt.subplots(figsize=(7.5, 6))

    if np.any(mask_low):
        plot_3d_scatter(ax, samples[mask_low], data[mask_low], color_low, "Adsorption <= 100")

    if np.any(mask_high):
        plot_3d_scatter(ax, samples[mask_high], data[mask_high], color_high, "Adsorption > 100")

    ax.set_xlabel("Samples", fontsize=24, labelpad=4)
    ax.set_ylabel(f"{col_name} Adsorption", fontsize=24, labelpad=6)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14,
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

    file_name = f"scatter_{col_name.split()[0]}_Times.svg"
    plt.savefig(file_name, format="svg")
    print(f"图表已保存: {file_name}")

    plt.show()
