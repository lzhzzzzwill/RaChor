import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# Global plotting style
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data.xlsx")
df = pd.read_excel(data_path)

x = df["SelectedMaxFeatures"]
series = [
    {
        "y": df["Sn MultiTaskLassoCV"],
        "label": "Sn MultiTaskLassoCV",
        "color": "#1f4e79",
        "linestyle": "-",
        "marker": "o",
        "text_offset": (-6, 16),
        "text_ha": "center",
        "text_va": "bottom",
    },
    {
        "y": df["Bi MultiTaskLassoCV"],
        "label": "Bi MultiTaskLassoCV",
        "color": "#b45f06",
        "linestyle": "-",
        "marker": "s",
        "text_offset": (0, 14),
        "text_ha": "center",
        "text_va": "bottom",
    },
    {
        "y": df["Sn Lasso"],
        "label": "Sn Lasso",
        "color": "#7f9bb3",
        "linestyle": "--",
        "marker": "^",
        "text_offset": (0, -16),
        "text_ha": "center",
        "text_va": "top",
    },
    {
        "y": df["Bi Lasso"],
        "label": "Bi Lasso",
        "color": "#d19a66",
        "linestyle": "--",
        "marker": "D",
        "text_offset": (14, 12),
        "text_ha": "left",
        "text_va": "bottom",
    },
]


fig, ax = plt.subplots(figsize=(17.28, 4.6))
ax.set_facecolor("white")

for item in series:
    ax.plot(
        x,
        item["y"],
        label=item["label"],
        color=item["color"],
        linestyle=item["linestyle"],
        linewidth=2.2,
        marker=item["marker"],
        markersize=4.6,
        markevery=6,
        markerfacecolor="white",
        markeredgewidth=0.9,
        alpha=0.97,
    )

    max_idx = item["y"].idxmax()
    max_x = x.loc[max_idx]
    max_y = item["y"].loc[max_idx]

    ax.scatter(
        max_x,
        max_y,
        s=145,
        facecolor="none",
        edgecolor=item["color"],
        linewidth=1.9,
        marker=item["marker"],
        zorder=6,
    )
    ax.scatter(
        max_x,
        max_y,
        s=58,
        facecolor=item["color"],
        edgecolor="white",
        linewidth=0.8,
        marker=item["marker"],
        zorder=7,
    )
    ax.annotate(
        f"{max_y:.3f}",
        xy=(max_x, max_y),
        xytext=item["text_offset"],
        textcoords="offset points",
        ha=item["text_ha"],
        va=item["text_va"],
        fontsize=10.2,
        color=item["color"],
        arrowprops={
            "arrowstyle": "-",
            "color": item["color"],
            "lw": 0.9,
            "shrinkA": 4,
            "shrinkB": 4,
            "alpha": 0.9,
        },
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        zorder=8,
    )


ax.set_xlabel("Number of Selected Features", fontsize=26, fontweight="normal")
ax.set_ylabel("OOF $R^2$", fontsize=26, fontweight="normal")

x_min, x_max = int(x.min()), int(x.max())
y_min = min(item["y"].min() for item in series)
y_max = max(item["y"].max() for item in series)
y_margin = max((y_max - y_min) * 0.12, 0.01)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min - y_margin, y_max + y_margin)
ax.margins(x=0.01)

ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

ax.tick_params(
    axis="both",
    which="major",
    labelsize=22,
    width=1.0,
    length=5,
    direction="out",
    color="#4d4d4d",
)
ax.tick_params(axis="y", which="minor", length=3, width=0.8, color="#7f7f7f")

ax.grid(axis="y", which="major", linestyle=(0, (3, 3)), linewidth=0.8, color="#d8d8d8")
ax.grid(False, axis="x")

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color("black")

ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="none",
    framealpha=0.82,
    fontsize=16,
    handlelength=2.6,
    columnspacing=1.5,
    handletextpad=0.6,
)

plt.tight_layout()

output_filename = "feature_selection_comparison.svg"
plt.savefig(output_filename, format="svg")
plt.show()

print(f"Figure saved as {output_filename}")
