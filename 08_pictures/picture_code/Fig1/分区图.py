from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. Spectral zone configuration
# ==========================================
ZONES = [
    {"name": "LFR", "range": (400, 900), "color": "#D8E7F5"},
    {"name": "MFR", "range": (900, 1400), "color": "#DDEDD8"},
    {"name": "HFR", "range": (1400, 1800), "color": "#FFF1C7"},
]
ZONE_ALPHA = 0.34


# ==========================================
# 2. Data
# ==========================================
file_path = Path(__file__).resolve().parent / "filter_snv_sg_scaled.csv"

try:
    df = pd.read_csv(file_path, header=0)
    x_raw = df.columns[1:708].astype(float).to_numpy()
    y_raw = df.iloc[1, 1:708].to_numpy(dtype=float)
except Exception:
    x_raw = np.linspace(400, 1800, 1400)
    y_raw = np.sin(x_raw / 150) * np.exp(-((x_raw - 1000) / 500) ** 2) + 0.5

y_raw = np.nan_to_num(y_raw)
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()


# ==========================================
# 3. Figure
# ==========================================
fig, ax = plt.subplots(figsize=(9, 4))

for zone in ZONES:
    start, end = zone["range"]
    ax.axvspan(start, end, facecolor=zone["color"], alpha=ZONE_ALPHA, lw=0, zorder=0)

ax.plot(x_raw, y_scaled, color="black", linewidth=1.5, zorder=2)


# ==========================================
# 4. Axes
# ==========================================
x_ticks = [400, 900, 1400, 1800]
ax.set_xlim(400, 1800)
ax.set_ylim(0, 1.2)
ax.set_xticks(x_ticks)
ax.set_yticks([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["left"].set_position(("data", 400))
ax.spines["bottom"].set_position(("data", 0))

ax.plot(400, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=8)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=8)


# ==========================================
# 5. Zone labels
# ==========================================
font_args = {"fontsize": 11, "fontweight": "bold", "ha": "center", "color": "#222222"}
text_y_pos = 1.1

for zone in ZONES:
    start, end = zone["range"]
    ax.text((start + end) / 2, text_y_pos, zone["name"], **font_args)

for boundary in [900, 1400]:
    ax.vlines(boundary, 0, 1.2, colors="white", linestyles="--", alpha=0.5, linewidth=1)

ax.tick_params(axis="x", labelsize=10)


# ==========================================
# 6. Save
# ==========================================
plt.tight_layout()
save_path = "spectrum_final_arrows.svg"
plt.savefig(save_path, format="svg", transparent=True, dpi=300)
print(f"Figure saved: {save_path}")

if "agg" not in matplotlib.get_backend().lower():
    plt.show()
